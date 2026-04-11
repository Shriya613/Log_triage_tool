# Surgical Log Triage Tool

An AI-assisted developer tool that ingests structured JSON device logs, validates the schema, detects anomalies and error patterns, and generates a plain-English triage summary for engineers.

Built with Python, FastAPI, and OpenAI. Designed to mirror real-world needs in safety-critical software (medical devices, robotics, navigation systems).

---

## Features

- REST API endpoint (`POST /triage`) accepting JSON log files
- Schema validation with clear error messages for malformed payloads
- Rule-based anomaly detection: repeated error codes, CRITICAL events, stuck states, component breakdowns
- LLM-generated triage summaries (GPT-4o-mini) with a rule-based fallback if no API key is set
- Full pytest suite: unit tests for parser + analyzer, integration tests for API

---

## Project Structure

```
log_triage/
├── main.py                   # FastAPI app and /triage endpoint
├── app/
│   ├── parser.py             # JSON schema validation and normalization
│   ├── analyzer.py           # Rule-based anomaly detection
│   └── summarizer.py         # LLM triage summary generation
├── tests/
│   └── test_triage.py        # Unit + integration tests (pytest)
├── sample_logs/
│   └── device_log_sample.json
├── requirements.txt
├── .env.example
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

# 2. Set your OpenAI API key (optional — falls back gracefully without it)
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Run the server
uvicorn main:app --reload

# 4. Try it out
curl -X POST http://localhost:8000/triage \
  -F "file=@sample_logs/device_log_sample.json"
```

API docs available at: `http://localhost:8000/docs`

---

## Running Tests

```bash
pytest tests/ -v
```

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
    "component_error_breakdown": {"calibration": 3, "motor": 1, "safety": 1}
  },
  "triage_summary": "The device log shows a HIGH risk profile with 2 critical failures and a repeated calibration error (code 3001, 3 occurrences). The motor controller became unresponsive followed by an emergency stop trigger in the safety component. Immediate inspection of the calibration and motor subsystems is recommended before next use."
}
```
