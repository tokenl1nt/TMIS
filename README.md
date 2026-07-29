# Trust the Mediator: Policy-Bounded LLM Mediation for Collaborative Intelligence Sharing

This directory contains the public benchmark data and evaluation scripts for
Trusted Model-Mediated Intelligence Sharing (TMSI).

## Contents

- `benchmark/` — scenarios, queries, and many-shot.
- `scripts/evaluate.py` — main and targeted benchmark evaluator.
- `scripts/manyprompt.py` — local Ollama many-shot evaluation.
- `scripts/copriva.py` — CoPriva benchmark adapter.
- `scripts/report.py` — Markdown report generation.
- `scripts/prompts.md` — System prompts used.

## Requirements

- Python 3.11 or newer.
- The dependencies listed in this directory's `pyproject.toml`.
- Credentials for the selected provider, such as `OPENAI_API_KEY` or
  `OPENROUTER_API_KEY`.

## Usage

Run commands from this directory:

```bash
python scripts/evaluate.py
python scripts/evaluate.py target --help
python scripts/manyprompt.py --help
python scripts/copriva.py --help
```
