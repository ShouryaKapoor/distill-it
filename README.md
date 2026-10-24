# distill-it

[![CI](https://github.com/ShouryaKapoor/distill-it/actions/workflows/ci.yml/badge.svg)](https://github.com/ShouryaKapoor/distill-it/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

> Prototype with an LLM. Ship a model that's 50× cheaper, runs on CPU, and only asks the LLM when it's unsure.

**Status: early development (pre-v0.1).** See [docs/DESIGN.md](docs/DESIGN.md) and [docs/PLAN.md](docs/PLAN.md).

## What it will do

```bash
pip install distill-it

distill-it run tickets.csv --labels billing,bug,feature_request,account \
    --teacher ollama/qwen2.5:7b --budget 400

distill-it serve runs/<id> --cascade --target-accuracy 0.95
```

1. Uses an LLM (any provider via LiteLLM) to label a small, actively chosen subset of your data.
2. Trains a SetFit student on those labels.
3. Calibrates it, exports it to ONNX, and serves it with an LLM fallback for low-confidence inputs.

## Development

```bash
uv sync                 # create .venv with dev dependencies
uv run pre-commit install
uv run pytest
uv run ruff check . && uv run ruff format .
```

## License

[Apache-2.0](LICENSE)
