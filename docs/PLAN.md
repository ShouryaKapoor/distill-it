# distill-it — Implementation Plan (≈14 days)

Legend: 🧑‍💻 = you write this piece (5–15 lines of core logic you should be able to explain in an interview) · ✅ = done-when criteria

---

## Phase 0 — Setup (Day 1)
- Install uv; init `pyproject.toml` (src layout), ruff, pytest, pre-commit
- Git repo, Apache-2.0 LICENSE, `.gitignore`, skeleton README
- GitHub Actions: lint + tests on push/PR

✅ `uv run pytest` passes on an empty test; CI green on GitHub.

## Phase 1 — Data + Teacher (Days 2–3)
- `data.py`: load CSV/JSONL/HF datasets → texts (+ gold labels if present)
- `teacher.py`: LiteLLM call, JSON-constrained label, retry → `abstain`, disk cache (JSONL)
- 🧑‍💻 Self-consistency aggregation: majority label + agreement score from `k` votes
- `FakeTeacher` for tests (returns gold label with configurable noise)
- Smoke test against Ollama locally and Gemini free tier

✅ `distill-it label data.csv --labels a,b,c` writes `labels.jsonl`; re-running uses the cache (0 new calls).

## Phase 2 — Student + Evaluation (Days 4–5)
- `student.py`: SetFit `fit`, `predict_proba`, `embed` (default body: MiniLM-L6)
- Metrics: accuracy, macro-F1, agreement, confusion matrix
- `benchmarks/`: dataset loaders for Banking77 / AG News / Emotion
- Colab notebook: start vLLM server → label benchmark pools → cache labels to Drive

✅ Student trained on 200 teacher labels evaluated against gold on all 3 datasets.

## Phase 3 — Active Learning (Days 6–7)
- `sampling.py`: sampler registry; `random`, `margin`, `entropy`
- 🧑‍💻 Margin score from a probability matrix (vectorised NumPy)
- 🧑‍💻 Hybrid selection: shortlist low-margin → k-means → nearest-to-centroid
- `loop.py`: cold start → rounds until budget; budget accounting; per-round metrics
- Benchmark: 3 strategies × 3 datasets × 5 budgets × 3 seeds (labels cached ⇒ cheap)

✅ Chart: accuracy vs. #LLM labels, hybrid ≥ random on ≥ 2/3 datasets (whatever the result, report it honestly).

## Phase 4 — Calibration, Cascade, Export (Days 8–9)
- `calibration.py`: fit temperature by NLL (scipy minimize); ECE metric
- 🧑‍💻 Threshold selection: lowest τ meeting target accuracy on the calibration split
- `cascade.py`: student-first predictor with teacher fallback; escalation stats
- `export.py`: ONNX export + int8 dynamic quantization; parity test (ONNX vs. PyTorch outputs)
- Latency benchmark: teacher vs. ONNX student on CPU (p50/p95)

✅ Chart: cascade accuracy vs. escalation rate; ECE drops after calibration; ONNX predictions match PyTorch.

## Phase 5 — CLI, Server, Docker (Days 10–11)
- `cli.py` (Typer): `run`, `label`, `train`, `eval`, `export`, `serve`, `benchmark`
- `serve.py`: FastAPI `/predict` (batch), `/health`, `/metrics`; optional cascade mode
- Dockerfile (slim, CPU onnxruntime), `docker run -p 8000:8000 ...` works
- `report.py`: results.json → PNG charts → single-file `report.html`

✅ Fresh clone → `uv sync` → `distill-it run examples/tickets.csv ...` → working server in < 10 min.

## Phase 6 — Demo, Docs, Launch (Days 12–14)
- Gradio app on HF Spaces with a pre-distilled Banking77 student (shows confidence + "would escalate")
- README: hero GIF, 30-second quickstart, benchmark table + charts, how it works, comparison to alternatives
- `CONTRIBUTING.md`, issue templates, 8–10 labelled `good first issue`s (new sampler, DistilBERT student, conformal cascade, multi-label, NER, W&B hook…)
- Tag `v0.1.0` → PyPI via trusted publishing
- Launch: LinkedIn post, r/MachineLearning / r/LocalLLaMA, Hacker News "Show HN"

✅ `pip install distill-it` works; demo live; resume bullet written with real numbers.

---

## Buffer and stretch (only if ahead)
- Conformal prediction cascade (risk-controlled escalation)
- DistilBERT student backend for large budgets
- Short write-up (blog/arXiv-style PDF) of the benchmark: useful for MSc and interviews
