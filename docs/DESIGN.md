# distill-it — Design Document

> Prototype with an LLM. Ship a model that's 50× cheaper, runs on CPU, and only asks the LLM when it's unsure.

## 1. Problem

Teams prototype text classification (support-ticket routing, intent detection, moderation) by prompting an LLM. It works on day one, then hits three walls in production:

| Wall | Why it hurts |
|---|---|
| **Cost** | Every prediction is an API call, so cost grows linearly with traffic |
| **Latency** | Hundreds of ms per call vs. single-digit ms for a small local model |
| **Privacy** | Customer text leaves your infrastructure (GDPR) |

Labelling data by hand to train a small model is slow and expensive. **distill-it** uses the LLM as the labeller and trains a small model from it, sending as few samples to the LLM as possible.

## 2. User experience (the north star)

```bash
pip install distill-it

distill-it run tickets.csv --labels billing,bug,feature_request,account \
    --teacher ollama/qwen2.5:7b --budget 400
# → labels ~400 samples via active learning
# → trains a SetFit student, calibrates it, exports ONNX
# → writes runs/<id>/report.html (accuracy, cost, latency, cascade curve)

distill-it serve runs/<id> --cascade --target-accuracy 0.95
# → FastAPI on :8000, student answers; unsure cases escalate to the teacher
```

## 3. Architecture

```
            ┌──────────────────────────── ACTIVE LEARNING LOOP ───────────────────────────┐
            │                                                                             │
 unlabeled  │   ┌──────────┐   samples   ┌──────────┐   labels    ┌──────────┐            │
 pool  ─────┼──►│ Sampler  │────────────►│ Teacher  │────────────►│ Student  │──┐         │
            │   │(sampling)│             │ (LiteLLM)│             │ (SetFit) │  │         │
            │   └────▲─────┘             └──────────┘             └──────────┘  │         │
            │        │          probabilities + embeddings on pool              │         │
            │        └──────────────────────────────────────────────────────────┘         │
            └───────────────────────────────────── until budget spent ────────────────────┘
                                                 │
                                                 ▼
                       ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐
                       │ Calibrator  │──►│  Exporter   │──►│ Cascade / Server    │
                       │ (temp. scl.)│   │ (ONNX int8) │   │ (FastAPI + fallback)│
                       └─────────────┘   └─────────────┘   └─────────────────────┘
                                                 │
                                                 ▼
                                          Report (JSON + HTML + charts)
```

### Module layout

```
src/distill_it/
├── config.py        # Pydantic run config (labels, budget, teacher, batch size…)
├── data.py          # Load CSV/JSONL/HF datasets → list[str] (+ optional gold labels)
├── teacher.py       # LiteLLM wrapper: structured JSON labels, optional self-consistency voting
├── student.py       # SetFit wrapper: fit / predict_proba / embed
├── sampling.py      # Strategies: random, margin, entropy, hybrid (margin + k-means diversity)
├── loop.py          # Active-learning loop orchestration, budget accounting
├── calibration.py   # Temperature scaling + threshold selection for a target accuracy
├── cascade.py       # Student-first predictor that escalates low-confidence inputs
├── export.py        # ONNX export + dynamic int8 quantization
├── serve.py         # FastAPI app (/predict, /health, /metrics)
├── report.py        # results.json → matplotlib charts → report.html
└── cli.py           # Typer CLI: run, label, train, eval, export, serve, benchmark
benchmarks/          # Reproducible benchmark scripts + Colab notebook (vLLM teacher)
demo/                # Gradio app for Hugging Face Spaces
tests/               # pytest, with a fake teacher so tests never call an LLM
```

**Plugin slots** (these become the contribution surface): `Teacher` providers (via LiteLLM), `Sampler` strategies (registry), `Student` backends (SetFit now, DistilBERT later), tasks (classification now; multi-label/NER later).

## 4. Key decisions

| Area | Decision | Alternatives considered | Why |
|---|---|---|---|
| LLM access | **LiteLLM** | Own provider classes; single SDK | One interface for OpenAI/Claude/Gemini/Ollama/vLLM; forkers use whatever they have |
| Student | **SetFit** (sentence-transformer + head) | DistilBERT fine-tune | Strong with 8–64 labels/class; trains on CPU in minutes, so it suits active learning's frequent retrains |
| Active learning | **Hybrid: margin uncertainty + k-means diversity** | Margin only; entropy | Avoids redundant batches; gives a clean ablation (random vs. margin vs. hybrid) |
| Teacher labels | **Constrained JSON + optional `--votes k` self-consistency** | Always vote; single zero-shot | Cheap by default; voting flags/drops samples the teacher disagrees with itself on |
| Cascade | **Temperature scaling + auto-threshold for a target accuracy** | Conformal prediction; raw softmax | Calibrated confidences make the threshold meaningful; easy to explain. Conformal = stretch goal |
| Benchmark teacher | **Open-weights model via vLLM on Colab/Kaggle** | API models; Ollama | Every README number reproducible for free; vLLM batching is ~10× faster than Ollama |
| Benchmark data | **Banking77, AG News, Emotion** | SST-2 | Hard / easy / noisy, which covers the spectrum |
| Tracking | **Built-in results.json + charts + HTML** | W&B; MLflow | Zero accounts needed; forks work immediately |
| Serving | **FastAPI + ONNX Runtime + Docker** | BentoML; Python-only | Industry standard; every layer is explainable in interviews |
| Demo | **Gradio on Hugging Face Spaces** | Streamlit; none | Free, clickable by recruiters, native to the ML community |
| Tooling | **uv, ruff, pytest, GitHub Actions, PyPI trusted publishing** | pip/conda | Modern, fast, one `pyproject.toml` |
| CLI / config | Typer + Pydantic | Click / argparse | Type-hint driven, auto `--help` |
| License | **Apache-2.0** | MIT | Patent grant; standard for ML tooling |

## 5. Algorithms

### 5.1 Hybrid active-learning sampler
1. Score every unlabeled sample with the current student → class probabilities `p`.
2. **Margin** = `p_top1 − p_top2` (small margin means the student is torn between two classes).
3. Keep the `5 × batch_size` lowest-margin candidates.
4. Run k-means (`k = batch_size`) on their embeddings; pick the sample nearest each centroid.
5. Send those to the teacher. Cold start (no student yet): k-means over the whole pool.

### 5.2 Teacher labelling
- Prompt includes the label set (+ optional per-label descriptions and few-shot examples); output is forced to JSON `{"label": <one of labels>}`.
- Invalid / off-set answers are retried once, then marked `abstain`.
- `--votes k`: sample `k` answers at temperature > 0; keep the majority label, record agreement `a ∈ [0,1]`; drop samples below `--min-agreement`.

### 5.3 Calibration and cascade
- Hold out a calibration split of teacher-labelled data.
- Fit a single temperature `T` minimizing NLL on that split: `p = softmax(log(p_raw) / T)`.
- For a target accuracy (relative to the teacher), sweep thresholds `τ`; choose the **lowest τ** whose accepted subset meets the target, which minimizes escalations.
- At inference: `max(p) ≥ τ` → student answers; else → teacher.

## 6. Evaluation

Per dataset, per strategy, at several label budgets (e.g. 50, 100, 200, 400, 800):

| Metric | Meaning |
|---|---|
| Teacher acc. vs gold | How good the LLM labeller is |
| Student acc. / macro-F1 vs gold | What you actually ship |
| Student–teacher agreement | How much distillation lost |
| LLM calls used | Labelling cost |
| Cost / 1k predictions | Teacher tokens × reference price vs. student (≈0) |
| p50 / p95 latency | Teacher vs. ONNX student on CPU |
| Escalation rate @ target | Fraction of traffic the cascade sends to the LLM |
| ECE (before/after calibration) | Does temperature scaling work? |

Headline charts: **accuracy vs. # LLM labels** (per strategy, mean ± std over 3 seeds) and **cascade accuracy vs. escalation rate**.

Cost honesty: the benchmark teacher is free (self-hosted), so cost is reported as **tokens × a configurable reference API price**, stated explicitly in the README.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Open 7B teacher weak on Banking77 (77 classes) | Report teacher-vs-gold separately; label descriptions in prompt; voting |
| Free Colab GPU limits / disconnects | Cache all teacher labels to disk (JSONL) so runs resume |
| SetFit + sklearn head ONNX export quirks | Use `setfit`'s ONNX exporter / skl2onnx; fallback: export body only and run head in NumPy |
| Retraining every AL round is slow | Small default body (MiniLM), fewer contrastive pairs per round, full retrain only at end |
| Scope creep | Conformal prediction, DistilBERT student, NER = post-v0.1 issues for contributors |
