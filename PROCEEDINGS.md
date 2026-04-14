# Project Proceedings — Surgical Log Triage Tool

A running log of decisions, progress, and next steps. Updated as the project evolves.

---

## Project Goal

Build a Python/FastAPI triage tool for medtech companies that:
- Ingests structured JSON device/system logs
- Validates and parses them with a pandas-based parser
- Detects anomalies and error patterns
- Generates a plain-English triage summary via a fine-tuned LLM
- Publishes the fine-tuned model to HuggingFace Hub

**HuggingFace account:** [Shriya613](https://huggingface.co/Shriya613)
**Target model repo:** `Shriya613/surgical-log-triage`

---

## Session 1 — 2026-04-11/12

### Starting State
- Files existed at root level: `main.py`, `parser.py`, `analyzer.py`, `summarizer.py`, `test_triage.py`
- `main.py` was importing from `app.parser`, `app.analyzer`, `app.summarizer` — but no `app/` directory existed
- `pandas` and `python-dotenv` were missing from `requirements.txt`
- `summarizer.py` was wired to OpenAI (`gpt-4o-mini`) — decision made to replace with a fine-tuned HuggingFace model instead

### Decisions Made
| Decision | Reasoning |
|---|---|
| Use `google/flan-t5-base` as the base model for fine-tuning | Small (250MB), seq2seq architecture suited for summarization, can run on CPU and Colab GPU |
| Fine-tune rather than use off-the-shelf model | Enables a genuine HuggingFace contribution; domain-specific output for medtech log triage |
| Write code as `.py` scripts, not Jupyter notebooks | Keeps code in the repo; run on Colab via `!python train.py` |
| Use Google Colab (free GPU tier) for fine-tuning | User has access; Lightning AI flagged as alternative |

### Completed
- [x] **Fix 2: Replaced OpenAI with HuggingFace local model**
  - Rewrote `app/summarizer.py` — loads `Shriya613/surgical-log-triage` first, falls back to `google/flan-t5-base`, then rule-based summary
  - Model loaded once at startup via `lru_cache`, reused across requests
  - `flagged_events` stripped from prompt to stay within flan-t5's 512-token input limit
  - Added `transformers`, `torch`, `sentencepiece`, `accelerate` to `requirements.txt`
  - OpenAI dependency removed entirely
  - Verified: **18/18 tests passing**

- [x] **Fix 1: App package structure**
  - Created `app/` directory with `__init__.py`
  - Moved `parser.py`, `analyzer.py`, `summarizer.py` into `app/`
  - Added `load_dotenv()` to `main.py`
  - Added `pandas==2.2.2` and `python-dotenv==1.0.1` to `requirements.txt`
  - Verified: **18/18 tests passing**

### Up Next
- [x] **Fix 2:** Rewrite `app/summarizer.py` to use a local HuggingFace model (`flan-t5-base`) instead of OpenAI
- [x] **Fix 3:** Write `train.py` — generates 230 synthetic training pairs, fine-tunes `flan-t5-base` via `Seq2SeqTrainer`, saves to `model_output/final/`, optional `--push` to HuggingFace Hub
- [ ] **Fix 4:** Push fine-tuned model to `Shriya613/surgical-log-triage` on HuggingFace Hub *(on hold — see training run below)*
- [ ] **Fix 5:** Add pandas to `app/parser.py` for richer analysis (time-series, drift detection)
- [ ] **Fix 6:** Run full end-to-end test with real sample log and verify triage summary quality

---

## Training Runs

| Run | Date | Base Model | Epochs | Train Loss | Published |
|---|---|---|---|---|---|
| [run_2026-04-13](results/run_2026-04-13.json) | 2026-04-13 | flan-t5-base | 5 | 6.929 | No |
| [run_2026-04-13b](results/run_2026-04-13b.json) | 2026-04-13 | flan-t5-base | 5 | 0 / nan ❌ | No |

**Target:** train_loss < 2.0, eval_loss stable before publishing to `Shriya613/surgical-log-triage`.
Run 13b failed due to fp16 instability (train_loss=0, eval_loss=nan) — fp16 disabled for next run.
Missing keys warning on embed_tokens.weight is expected for flan-t5 tied weights — not an error.

---

## Open Questions / To Revisit
- Do we want a `tests/` subdirectory or keep `test_triage.py` at root? (CLAUDE.md mentions `pytest tests/`)
- Should the API support batch log uploads (multiple devices at once)?
- What synthetic training data format works best for `flan-t5` fine-tuning?