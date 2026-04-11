"""
Tests for parser.py, analyzer.py, and the /triage API endpoint.
Run with: pytest tests/ -v
"""

import json
import pytest
from fastapi.testclient import TestClient
from io import BytesIO

from main import app
from app.parser import parse_log
from app.analyzer import analyze_log

client = TestClient(app)

# ── Fixtures ──────────────────────────────────────────────────────────────────

VALID_LOG = {
    "device_id": "DEV-001",
    "timestamp": "2024-11-01T10:00:00Z",
    "firmware_version": "2.3.1",
    "events": [
        {"event_id": "E001", "timestamp": "2024-11-01T10:00:01Z", "level": "INFO", "message": "System startup", "code": "1000", "component": "core"},
        {"event_id": "E002", "timestamp": "2024-11-01T10:00:05Z", "level": "WARNING", "message": "Sensor drift detected", "code": "2001", "component": "sensor"},
        {"event_id": "E003", "timestamp": "2024-11-01T10:00:10Z", "level": "ERROR", "message": "Calibration failed", "code": "3001", "component": "calibration"},
        {"event_id": "E004", "timestamp": "2024-11-01T10:00:15Z", "level": "ERROR", "message": "Calibration failed", "code": "3001", "component": "calibration"},
        {"event_id": "E005", "timestamp": "2024-11-01T10:00:20Z", "level": "ERROR", "message": "Calibration failed", "code": "3001", "component": "calibration"},
        {"event_id": "E006", "timestamp": "2024-11-01T10:00:25Z", "level": "CRITICAL", "message": "Motor controller unresponsive", "code": "5001", "component": "motor"},
    ]
}

HIGH_RISK_LOG = {
    "device_id": "DEV-002",
    "timestamp": "2024-11-01T11:00:00Z",
    "events": [
        {"event_id": f"E{i:03}", "timestamp": "2024-11-01T11:00:01Z", "level": "CRITICAL", "message": "Power failure", "code": "9001", "component": "power"}
        for i in range(5)
    ]
}


# ── Parser tests ──────────────────────────────────────────────────────────────

class TestParser:

    def test_valid_log_parses_correctly(self):
        result = parse_log(VALID_LOG)
        assert result["device_id"] == "DEV-001"
        assert len(result["events"]) == 6
        assert result["firmware_version"] == "2.3.1"

    def test_missing_top_level_field_raises(self):
        bad = {k: v for k, v in VALID_LOG.items() if k != "events"}
        with pytest.raises(ValueError, match="Missing required top-level fields"):
            parse_log(bad)

    def test_empty_events_raises(self):
        bad = {**VALID_LOG, "events": []}
        with pytest.raises(ValueError, match="empty"):
            parse_log(bad)

    def test_invalid_level_raises(self):
        bad_event = {**VALID_LOG["events"][0], "level": "VERBOSE"}
        bad = {**VALID_LOG, "events": [bad_event]}
        with pytest.raises(ValueError, match="invalid level"):
            parse_log(bad)

    def test_missing_event_field_raises(self):
        bad_event = {k: v for k, v in VALID_LOG["events"][0].items() if k != "message"}
        bad = {**VALID_LOG, "events": [bad_event]}
        with pytest.raises(ValueError, match="missing fields"):
            parse_log(bad)

    def test_level_normalized_to_uppercase(self):
        lower_event = {**VALID_LOG["events"][0], "level": "error"}
        log = {**VALID_LOG, "events": [lower_event]}
        result = parse_log(log)
        assert result["events"][0]["level"] == "ERROR"

    def test_optional_fields_default_gracefully(self):
        minimal_event = {
            "event_id": "E001", "timestamp": "2024-11-01T10:00:01Z",
            "level": "INFO", "message": "Startup"
        }
        log = {**VALID_LOG, "events": [minimal_event]}
        result = parse_log(log)
        assert result["events"][0]["code"] == "N/A"
        assert result["events"][0]["component"] == "unknown"


# ── Analyzer tests ─────────────────────────────────────────────────────────────

class TestAnalyzer:

    def test_high_risk_on_critical_events(self):
        parsed = parse_log(HIGH_RISK_LOG)
        findings = analyze_log(parsed)
        assert findings["risk_level"] == "HIGH"

    def test_repeated_error_codes_detected(self):
        parsed = parse_log(VALID_LOG)
        findings = analyze_log(parsed)
        assert "3001" in findings["repeated_error_codes"]
        assert findings["repeated_error_codes"]["3001"] == 3

    def test_critical_events_isolated(self):
        parsed = parse_log(VALID_LOG)
        findings = analyze_log(parsed)
        assert len(findings["critical_events"]) == 1
        assert findings["critical_events"][0]["code"] == "5001"

    def test_component_breakdown_populated(self):
        parsed = parse_log(VALID_LOG)
        findings = analyze_log(parsed)
        assert "calibration" in findings["component_error_breakdown"]

    def test_nominal_risk_on_clean_log(self):
        clean_log = {
            "device_id": "DEV-CLEAN",
            "timestamp": "2024-11-01T09:00:00Z",
            "events": [
                {"event_id": "E001", "timestamp": "2024-11-01T09:00:01Z",
                 "level": "INFO", "message": "All systems nominal", "code": "1000", "component": "core"}
            ]
        }
        parsed = parse_log(clean_log)
        findings = analyze_log(parsed)
        assert findings["risk_level"] == "NOMINAL"


# ── API endpoint tests ─────────────────────────────────────────────────────────

class TestTriageEndpoint:

    def _upload(self, data: dict):
        content = json.dumps(data).encode()
        return client.post(
            "/triage",
            files={"file": ("test_log.json", BytesIO(content), "application/json")}
        )

    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_valid_log_returns_200(self):
        response = self._upload(VALID_LOG)
        assert response.status_code == 200
        body = response.json()
        assert body["device_id"] == "DEV-001"
        assert "triage_summary" in body
        assert "findings" in body

    def test_invalid_json_returns_422(self):
        response = client.post(
            "/triage",
            files={"file": ("bad.json", BytesIO(b"{not valid json"), "application/json")}
        )
        assert response.status_code == 422

    def test_non_json_file_returns_400(self):
        response = client.post(
            "/triage",
            files={"file": ("log.txt", BytesIO(b"some text"), "text/plain")}
        )
        assert response.status_code == 400

    def test_missing_required_field_returns_422(self):
        bad = {k: v for k, v in VALID_LOG.items() if k != "device_id"}
        response = self._upload(bad)
        assert response.status_code == 422

    def test_total_events_count_correct(self):
        response = self._upload(VALID_LOG)
        assert response.json()["total_events"] == len(VALID_LOG["events"])
