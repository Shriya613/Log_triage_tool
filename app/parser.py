"""
parser.py
Validates incoming JSON log payloads with soft validation:
  - Collects ALL schema issues instead of raising on the first error
  - Returns schema_valid=False + populated schema_issues on bad input
  - Always returns a result — even malformed logs produce a best-effort parse
  - Runs pandas time-series analysis on whatever valid events are present
"""

import pandas as pd
from app.schemas import VALID_LEVELS

REQUIRED_TOP_LEVEL    = {"device_id", "timestamp", "events"}
REQUIRED_EVENT_FIELDS = {"event_id", "timestamp", "level", "message"}

# Errors firing faster than this (seconds avg interval) are flagged as rapid-repeat
RAPID_REPEAT_THRESHOLD_SECONDS = 10
# Numeric values beyond this magnitude are flagged as extreme
EXTREME_VALUE_THRESHOLD = 1000.0


def parse_log(data: dict) -> dict:
    """
    Soft-validate and parse a raw JSON log payload.

    Returns:
        schema_valid (bool)       — True only if zero issues found
        schema_issues (list[str]) — all validation problems, human-readable
        device_id, timestamp, firmware_version
        events (list[dict])       — best-effort validated events
        time_analysis (dict)      — pandas-derived metrics
    """
    issues: list[str] = []

    # ── Top-level required fields ──────────────────────────────────────────────
    missing_top = REQUIRED_TOP_LEVEL - data.keys()
    for field in sorted(missing_top):
        issues.append(f"Missing required top-level field: '{field}'")

    if "events" not in data:
        return _empty_result(issues)

    if not isinstance(data["events"], list):
        issues.append("'events' must be a list")
        return _empty_result(issues)

    if len(data["events"]) == 0:
        issues.append("'events' list is empty — nothing to triage")
        return _empty_result(issues)

    # ── Per-event validation ────────────────────────────────────────────────────
    validated_events = []
    for i, event in enumerate(data["events"]):
        if not isinstance(event, dict):
            issues.append(f"Event[{i}]: not a valid object (got {type(event).__name__})")
            continue

        # Missing required fields
        for field in sorted(REQUIRED_EVENT_FIELDS - event.keys()):
            issues.append(f"Event[{i}]: missing required field '{field}'")

        # Level validation
        if "level" in event:
            level = str(event["level"]).upper()
            if level not in VALID_LEVELS:
                issues.append(
                    f"Event[{i}]: invalid level '{event['level']}' "
                    f"(expected one of {sorted(VALID_LEVELS)})"
                )

        # Numeric value type check
        if "value" in event and event["value"] is not None:
            try:
                float(event["value"])
            except (TypeError, ValueError):
                issues.append(
                    f"Event[{i}]: 'value' must be numeric, "
                    f"got {type(event['value']).__name__} ('{event['value']}')"
                )

        # Extreme value check
        if "value" in event and event["value"] is not None:
            try:
                v = float(event["value"])
                if abs(v) > EXTREME_VALUE_THRESHOLD:
                    issues.append(
                        f"Event[{i}]: extreme sensor value {v} "
                        f"(threshold ±{EXTREME_VALUE_THRESHOLD})"
                    )
            except (TypeError, ValueError):
                pass

        # Best-effort normalisation — use whatever fields are present
        validated_events.append({
            "event_id":  str(event.get("event_id", f"unknown_{i}")),
            "timestamp": str(event.get("timestamp", "")),
            "level":     str(event.get("level", "INFO")).upper(),
            "message":   str(event.get("message", "")),
            "code":      str(event.get("code", "N/A")),
            "component": str(event.get("component", "unknown")),
            "value":     event.get("value"),
        })

    # ── Pandas time analysis ────────────────────────────────────────────────────
    time_analysis = _analyse_timestamps(validated_events) if validated_events else {}

    return {
        "schema_valid":     len(issues) == 0,
        "schema_issues":    issues,
        "device_id":        str(data.get("device_id", "UNKNOWN")),
        "timestamp":        str(data.get("timestamp", "")),
        "firmware_version": str(data.get("firmware_version", "unknown")),
        "events":           validated_events,
        "time_analysis":    time_analysis,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _empty_result(issues: list[str]) -> dict:
    """Return a minimal failed-parse result when events cannot be processed."""
    return {
        "schema_valid":     False,
        "schema_issues":    issues,
        "device_id":        "UNKNOWN",
        "timestamp":        "",
        "firmware_version": "unknown",
        "events":           [],
        "time_analysis":    {},
    }


def _analyse_timestamps(events: list[dict]) -> dict:
    """
    Build a DataFrame from validated events and compute time-based metrics.
    Unparseable timestamps become NaT and are excluded from calculations.
    """
    df = pd.DataFrame(events)
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    parseable = df.dropna(subset=["ts"]).sort_values("ts")

    if parseable.empty:
        return {"note": "No parseable timestamps — time analysis unavailable."}

    first_ts   = parseable["ts"].iloc[0]
    last_ts    = parseable["ts"].iloc[-1]
    duration_s = (last_ts - first_ts).total_seconds()
    rate_per_min = (len(parseable) / duration_s * 60) if duration_s > 0 else 0.0
    out_of_order = int(
        (parseable["ts"].diff().dt.total_seconds().dropna() < 0).sum()
    )
    rapid = _detect_rapid_repeats(parseable)

    return {
        "log_duration_seconds":  round(duration_s, 2),
        "event_rate_per_minute": round(rate_per_min, 2),
        "out_of_order_events":   out_of_order,
        "rapid_repeat_errors":   rapid,
    }


def _detect_rapid_repeats(df: pd.DataFrame) -> list[dict]:
    """
    Return error/critical codes whose average firing interval is below the
    RAPID_REPEAT_THRESHOLD_SECONDS — indicates a tight loop or stuck retry.
    """
    error_df = df[df["level"].isin({"ERROR", "CRITICAL"}) & (df["code"] != "N/A")]
    rapid = []
    for code, group in error_df.groupby("code"):
        if len(group) < 2:
            continue
        intervals = group["ts"].sort_values().diff().dt.total_seconds().dropna()
        avg_interval = intervals.mean()
        if avg_interval < RAPID_REPEAT_THRESHOLD_SECONDS:
            rapid.append({
                "code":                 code,
                "component":            group["component"].iloc[0],
                "occurrences":          len(group),
                "avg_interval_seconds": round(avg_interval, 2),
            })
    return rapid