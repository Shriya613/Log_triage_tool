"""
schemas.py
Pydantic models for the triage pipeline's input/output contracts.
Used for validation in parser.py and response typing in main.py.
"""

from pydantic import BaseModel
from typing import Literal

VALID_LEVELS = {"INFO", "WARNING", "ERROR", "CRITICAL"}

# Anomaly types produced by analyzer.py and used in training data generators
ANOMALY_TYPES = {
    "critical_event",    # CRITICAL level event
    "repeated_error",    # same error code fires 3+ times
    "warning_cluster",   # elevated warning volume (low severity indicator)
    "stuck_state",       # same message repeats 3+ times
    "rapid_repeat",      # error fires multiple times within threshold seconds
    "out_of_order",      # event timestamp is earlier than preceding event
    "extreme_value",     # numeric sensor reading exceeds ±EXTREME_VALUE_THRESHOLD
    "missing_field",     # required field absent (schema issue)
    "escalation",        # warn → error → critical progression in one session
}


class LogEvent(BaseModel):
    event_id: str
    timestamp: str
    level: str
    message: str
    code: str | None = None
    component: str | None = None
    value: float | None = None      # optional numeric sensor reading


class DeviceLog(BaseModel):
    device_id: str
    timestamp: str
    firmware_version: str | None = None
    events: list[dict]              # validated event-by-event inside parser.py


class Anomaly(BaseModel):
    type: str                       # one of ANOMALY_TYPES
    component: str
    detail: str
    code: str | None = None
    occurrences: int | None = None


class TriageOutput(BaseModel):
    schema_valid: bool
    schema_issues: list[str]
    anomalies: list[dict]
    severity: Literal["low", "medium", "high"]
    root_cause: str
    summary: str