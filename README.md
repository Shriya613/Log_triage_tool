# Surgical Log Triage Tool

An AI-assisted developer tool that ingests structured JSON device logs, validates the schema, detects anomalies and error patterns, and generates a plain-English triage summary for engineers.

Built with Python, FastAPI, and a fine-tuned HuggingFace model (`Shriya613/surgical-log-triage`). Designed for medtech and safety-critical software teams.

---

## Features

- REST API endpoint (`POST /triage`) accepting JSON log files
- Schema validation with clear error messages for malformed payloads
- Rule-based anomaly detection: repeated error codes, CRITICAL events, stuck states, component breakdowns
- Pandas-based time-series analysis: log duration, event rate, out-of-order timestamps, rapid-repeat error detection
- LLM-generated triage summaries via a fine-tuned `flan-t5-base` model, with a rule-based fallback
- Unit tests for parser + analyzer, integration tests for API

---

## Project Structure

```
log_triage_tool/
├── main.py                      # FastAPI app — /triage and /health endpoints
├── app/
│   ├── __init__.py
│   ├── parser.py                # JSON schema validation, normalization, time-series analysis
│   ├── analyzer.py              # Rule-based anomaly and risk-level detection
│   └── summarizer.py            # Loads fine-tuned HuggingFace model, generates triage summary
├── train.py                     # Fine-tuning script (run on Colab)
├── results/                     # Training run result JSONs
├── test_triage.py               # Unit + integration tests (pytest)
├── device_log_sample.json       # Sample input for manual testing
├── requirements.txt
└── .gitignore
```

---

## Quickstart

```bash
# 1. Clone and set up environment
git clone https://github.com/yourusername/log-triage-tool.git
cd log-triage-tool

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Run the server
uvicorn main:app --reload

# 3. Try it out
curl -X POST http://localhost:8000/triage \
  -F "file=@device_log_sample.json"
```

API docs available at: `http://localhost:8000/docs`

---

## Running Tests

```bash
pytest test_triage.py -v
```

---

## Model

The summarizer loads models in this order:

1. `Shriya613/surgical-log-triage` — fine-tuned on surgical device log triage data
2. `google/flan-t5-base` — base model fallback
3. Rule-based summary — if neither model is available

The model is loaded once at startup and reused across requests.

---

## Fine-Tuning (Colab)

Training uses `google/flan-t5-base` with synthetic data (500 examples, 7 scenario types) and optionally the [AI4I 2020 Predictive Maintenance dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020) from Kaggle.

```bash
# On Google Colab — after cloning the repo and uploading ai4i2020.csv to /content/

# Synthetic data only
!python train.py

# Synthetic + external dataset
!python train.py --extra-data /content/ai4i2020.csv

# Train and publish to HuggingFace Hub
!python train.py --extra-data /content/ai4i2020.csv --push --hub-token YOUR_TOKEN
```

Model output is saved to `model_output/final/`. Publish to HuggingFace once `train_loss < 2.0` and `eval_loss` is stable (not nan).

---

## Expected JSON Log Format

```json
{
  "device_id": "DEV-001",
  "timestamp": "2024-11-01T10:00:00Z",
  "firmware_version": "2.3.1",
  "events": [
    {
      "event_id": "E001",
      "timestamp": "2024-11-01T10:00:01Z",
      "level": "INFO",
      "message": "System startup complete",
      "code": "1000",
      "component": "core"
    }
  ]
}
```

**Required fields:** `device_id`, `timestamp`, `events`
**Required per event:** `event_id`, `timestamp`, `level`, `message`
**Valid levels:** `INFO`, `WARNING`, `ERROR`, `CRITICAL`
**Optional per event:** `code`, `component`

---

## Example Response

```json
{
  "device_id": "DEV-SURGICAL-001",
  "log_timestamp": "2024-11-01T10:00:00Z",
  "total_events": 9,
  "findings": {
    "risk_level": "HIGH",
    "level_counts": {"INFO": 2, "WARNING": 2, "ERROR": 3, "CRITICAL": 2},
    "total_flagged_events": 5,
    "repeated_error_codes": {"3001": 3},
    "component_error_breakdown": {"calibration": 3, "motor": 1, "safety": 1},
    "time_analysis": {
      "log_duration_seconds": 142.0,
      "event_rate_per_minute": 3.8,
      "out_of_order_events": 0,
      "rapid_repeat_errors": ["3001"]
    }
  },
  "triage_summary": "The device log shows a HIGH risk profile with 2 critical failures and a repeated calibration error (code 3001, 3 occurrences). The motor controller became unresponsive followed by an emergency stop trigger in the safety component. Immediate inspection of the calibration and motor subsystems is recommended before next use."
}
```
