"""
test_triage.py
Tests for parser.py, analyzer.py, and the /triage API endpoint.

Changes in v2:
  - parser.py now uses soft validation (no ValueError raises).
    Tests check schema_valid==False and schema_issues content instead.
  - analyzer.py returns anomalies (typed list) and severity (low|medium|high).
  - /triage returns strict JSON: schema_valid, schema_issues, anomalies, severity,
    root_cause, summary.

Run with: pytest test_triage.py -v
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
        {"event_id": "E001", "timestamp": "2024-11-01T10:00:01Z",
         "level": "INFO",     "message": "System startup",      "code": "1000", "component": "core"},
        {"event_id": "E002", "timestamp": "2024-11-01T10:00:05Z",
         "level": "WARNING",  "message": "Sensor drift",        "code": "2001", "component": "sensor"},
        {"event_id": "E003", "timestamp": "2024-11-01T10:00:10Z",
         "level": "ERROR",    "message": "Calibration failed",  "code": "3001", "component": "calibration"},
        {"event_id": "E004", "timestamp": "2024-11-01T10:00:15Z",
         "level": "ERROR",    "message": "Calibration failed",  "code": "3001", "component": "calibration"},
        {"event_id": "E005", "timestamp": "2024-11-01T10:00:20Z",
         "level": "ERROR",    "message": "Calibration failed",  "code": "3001", "component": "calibration"},
        {"event_id": "E006", "timestamp": "2024-11-01T10:00:25Z",
         "level": "CRITICAL", "message": "Motor controller unresponsive",
         "code": "5001", "component": "motor"},
    ],
}

HIGH_RISK_LOG = {
    "device_id": "DEV-002",
    "timestamp": "2024-11-01T11:00:00Z",
    "events": [
        {"event_id": f"E{i:03}", "timestamp": "2024-11-01T11:00:01Z",
         "level": "CRITICAL", "message": "Power failure", "code": "9001", "component": "power"}
        for i in range(5)
    ],
}

CLEAN_LOG = {
    "device_id": "DEV-CLEAN",
    "timestamp": "2024-11-01T09:00:00Z",
    "events": [
        {"event_id": "E001", "timestamp": "2024-11-01T09:00:01Z",
         "level": "INFO", "message": "All systems nominal", "code": "1000", "component": "core"},
    ],
}


# ── Parser tests ──────────────────────────────────────────────────────────────

class TestParser:

    def test_valid_log_schema_valid_true(self):
        result = parse_log(VALID_LOG)
        assert result["schema_valid"] is True
        assert result["schema_issues"] == []

    def test_valid_log_parses_correctly(self):
        result = parse_log(VALID_LOG)
        assert result["device_id"] == "DEV-001"
        assert len(result["events"]) == 6
        assert result["firmware_version"] == "2.3.1"

    def test_missing_top_level_field_schema_invalid(self):
        bad = {k: v for k, v in VALID_LOG.items() if k != "device_id"}
        result = parse_log(bad)
        assert result["schema_valid"] is False
        assert any("device_id" in issue for issue in result["schema_issues"])

    def test_empty_events_schema_invalid(self):
        bad = {**VALID_LOG, "events": []}
        result = parse_log(bad)
        assert result["schema_valid"] is False
        assert any("empty" in issue for issue in result["schema_issues"])

    def test_invalid_level_schema_invalid(self):
        bad_event = {**VALID_LOG["events"][0], "level": "VERBOSE"}
        bad = {**VALID_LOG, "events": [bad_event]}
        result = parse_log(bad)
        assert result["schema_valid"] is False
        assert any("VERBOSE" in issue for issue in result["schema_issues"])

    def test_missing_event_field_schema_invalid(self):
        bad_event = {k: v for k, v in VALID_LOG["events"][0].items() if k != "message"}
        bad = {**VALID_LOG, "events": [bad_event]}
        result = parse_log(bad)
        assert result["schema_valid"] is False
        assert any("message" in issue for issue in result["schema_issues"])

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

    def test_soft_validation_still_parses_events(self):
        """Even a log with schema issues should return parsed events (best-effort)."""
        bad = {k: v for k, v in VALID_LOG.items() if k != "device_id"}
        result = parse_log(bad)
        assert result["schema_valid"] is False
        assert len(result["events"]) == len(VALID_LOG["events"])  # events still parsed

    def test_time_analysis_present(self):
        result = parse_log(VALID_LOG)
        ta = result["time_analysis"]
        assert "log_duration_seconds" in ta
        assert "event_rate_per_minute" in ta
        assert "out_of_order_events" in ta
        assert "rapid_repeat_errors" in ta

    def test_rapid_repeat_detected(self):
        """3 identical error codes within 5s each → rapid_repeat."""
        log = {
            "device_id": "DEV-RR", "timestamp": "2024-01-01T00:00:00Z",
            "events": [
                {"event_id": f"E{i}", "timestamp": f"2024-01-01T00:00:0{i+1}Z",
                 "level": "ERROR", "message": "Fail", "code": "3001", "component": "calibration"}
                for i in range(3)
            ],
        }
        result = parse_log(log)
        assert len(result["time_analysis"]["rapid_repeat_errors"]) > 0
        assert result["time_analysis"]["rapid_repeat_errors"][0]["code"] == "3001"


# ── Analyzer tests ─────────────────────────────────────────────────────────────

class TestAnalyzer:

    def test_severity_is_lowercase_enum(self):
        parsed = parse_log(VALID_LOG)
        findings = analyze_log(parsed)
        assert findings["severity"] in {"low", "medium", "high"}

    def test_high_severity_on_critical_events(self):
        parsed = parse_log(HIGH_RISK_LOG)
        findings = analyze_log(parsed)
        assert findings["severity"] == "high"

    def test_anomalies_is_list(self):
        parsed = parse_log(VALID_LOG)
        findings = analyze_log(parsed)
        assert isinstance(findings["anomalies"], list)

    def test_critical_event_anomaly_detected(self):
        parsed = parse_log(VALID_LOG)
        findings = analyze_log(parsed)
        types = {a["type"] for a in findings["anomalies"]}
        assert "critical_event" in types

    def test_repeated_error_anomaly_detected(self):
        parsed = parse_log(VALID_LOG)
        findings = analyze_log(parsed)
        repeated = [a for a in findings["anomalies"] if a["type"] == "repeated_error"]
        assert len(repeated) > 0
        assert repeated[0]["code"] == "3001"
        assert repeated[0]["occurrences"] == 3

    def test_anomaly_has_required_fields(self):
        parsed = parse_log(VALID_LOG)
        findings = analyze_log(parsed)
        for a in findings["anomalies"]:
            assert "type" in a
            assert "component" in a
            assert "detail" in a

    def test_component_breakdown_populated(self):
        parsed = parse_log(VALID_LOG)
        findings = analyze_log(parsed)
        assert "calibration" in findings["component_error_breakdown"]

    def test_low_severity_on_clean_log(self):
        parsed = parse_log(CLEAN_LOG)
        findings = analyze_log(parsed)
        assert findings["severity"] == "low"
        assert findings["anomalies"] == []

    def test_missing_field_anomaly_from_schema_issues(self):
        """Schema issues (missing fields) should surface as missing_field anomalies."""
        bad = {k: v for k, v in VALID_LOG.items() if k != "device_id"}
        parsed   = parse_log(bad)
        findings = analyze_log(parsed)
        types = {a["type"] for a in findings["anomalies"]}
        assert "missing_field" in types

    def test_warning_cluster_anomaly(self):
        """5+ warnings → warning_cluster anomaly."""
        log = {
            "device_id": "DEV-WARN", "timestamp": "2024-01-01T00:00:00Z",
            "events": [
                {"event_id": f"E{i}", "timestamp": f"2024-01-01T00:00:{i+1:02d}Z",
                 "level": "WARNING", "message": "Drift", "code": "2001", "component": "sensor"}
                for i in range(6)
            ],
        }
        parsed   = parse_log(log)
        findings = analyze_log(parsed)
        types = {a["type"] for a in findings["anomalies"]}
        assert "warning_cluster" in types


# ── API endpoint tests ─────────────────────────────────────────────────────────

class TestTriageEndpoint:

    def _upload(self, data: dict):
        content = json.dumps(data).encode()
        return client.post(
            "/triage",
            files={"file": ("test_log.json", BytesIO(content), "application/json")},
        )

    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_valid_log_returns_200(self):
        response = self._upload(VALID_LOG)
        assert response.status_code == 200

    def test_response_has_strict_contract_fields(self):
        response = self._upload(VALID_LOG)
        body = response.json()
        for field in ("schema_valid", "schema_issues", "anomalies",
                      "severity", "root_cause", "summary"):
            assert field in body, f"Missing field in response: {field}"

    def test_schema_valid_true_for_good_log(self):
        body = self._upload(VALID_LOG).json()
        assert body["schema_valid"] is True
        assert body["schema_issues"] == []

    def test_schema_valid_false_for_bad_log(self):
        """Invalid schema → HTTP 200 with schema_valid=False, not a 422."""
        bad  = {k: v for k, v in VALID_LOG.items() if k != "device_id"}
        resp = self._upload(bad)
        assert resp.status_code == 200
        body = resp.json()
        assert body["schema_valid"] is False
        assert len(body["schema_issues"]) > 0

    def test_severity_is_valid_enum(self):
        body = self._upload(VALID_LOG).json()
        assert body["severity"] in {"low", "medium", "high"}

    def test_anomalies_is_list(self):
        body = self._upload(VALID_LOG).json()
        assert isinstance(body["anomalies"], list)

    def test_root_cause_and_summary_are_strings(self):
        body = self._upload(VALID_LOG).json()
        assert isinstance(body["root_cause"], str) and len(body["root_cause"]) > 0
        assert isinstance(body["summary"], str) and len(body["summary"]) > 0

    def test_invalid_json_returns_422(self):
        response = client.post(
            "/triage",
            files={"file": ("bad.json", BytesIO(b"{not valid json"), "application/json")},
        )
        assert response.status_code == 422

    def test_non_json_file_returns_400(self):
        response = client.post(
            "/triage",
            files={"file": ("log.txt", BytesIO(b"some text"), "text/plain")},
        )
        assert response.status_code == 400

    def test_total_events_count_correct(self):
        body = self._upload(VALID_LOG).json()
        assert body["total_events"] == len(VALID_LOG["events"])

    def test_high_risk_log_severity_is_high(self):
        body = self._upload(HIGH_RISK_LOG).json()
        assert body["severity"] == "high"