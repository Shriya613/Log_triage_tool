"""
parser.py
Validates incoming JSON log payloads against the expected schema,
then uses pandas to compute time-based metrics: log duration, event rate,
out-of-order timestamps, and rapid repeat errors.
Raises ValueError with a clear message if required fields are missing or malformed.
"""

import pandas as pd

REQUIRED_TOP_LEVEL   = {"device_id", "timestamp", "events"}
REQUIRED_EVENT_FIELDS = {"event_id", "timestamp", "level", "message"}
VALID_LEVELS         = {"INFO", "WARNING", "ERROR", "CRITICAL"}

# Errors repeating faster than this (seconds) are flagged as a rapid-repeat loop
RAPID_REPEAT_THRESHOLD_SECONDS = 10


def parse_log(data: dict) -> dict:
    """
    Validate, normalize, and time-analyse a raw JSON log payload.

    Args:
        data: Raw parsed JSON dict from the uploaded file.

    Returns:
        Validated log dict, extended with time-based metrics under
        the key 'time_analysis'.

    Raises:
        ValueError: If required fields are missing or values are invalid.
    """
    # ── Schema validation ──────────────────────────────────────────────────────
    missing = REQUIRED_TOP_LEVEL - data.keys()
    if missing:
        raise ValueError(f"Missing required top-level fields: {missing}")

    if not isinstance(data["events"], list):
        raise ValueError("'events' must be a list.")

    if len(data["events"]) == 0:
        raise ValueError("'events' list is empty. Nothing to triage.")

    validated_events = []
    for i, event in enumerate(data["events"]):
        if not isinstance(event, dict):
            raise ValueError(f"Event at index {i} is not a valid object.")

        missing_fields = REQUIRED_EVENT_FIELDS - event.keys()
        if missing_fields:
            raise ValueError(f"Event {i} missing fields: {missing_fields}")

        level = str(event["level"]).upper()
        if level not in VALID_LEVELS:
            raise ValueError(
                f"Event {i} has invalid level '{event['level']}'. "
                f"Must be one of: {VALID_LEVELS}"
            )

        validated_events.append({
            "event_id":  str(event["event_id"]),
            "timestamp": str(event["timestamp"]),
            "level":     level,
            "message":   str(event["message"]),
            "code":      str(event.get("code", "N/A")),
            "component": str(event.get("component", "unknown")),
        })

    # ── Pandas time analysis ───────────────────────────────────────────────────
    time_analysis = _analyse_timestamps(validated_events)

    return {
        "device_id":        str(data["device_id"]),
        "timestamp":        str(data["timestamp"]),
        "firmware_version": str(data.get("firmware_version", "unknown")),
        "events":           validated_events,
        "time_analysis":    time_analysis,
    }


def _analyse_timestamps(events: list[dict]) -> dict:
    """
    Build a DataFrame from validated events and compute time-based metrics.
    All timestamp parsing uses coerce so a single bad timestamp doesn't crash
    the whole log — unparseable entries become NaT and are excluded from maths.
    """
    df = pd.DataFrame(events)

    # Parse timestamps; bad values become NaT
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    parseable = df.dropna(subset=["ts"]).sort_values("ts")

    if parseable.empty:
        return {"note": "No parseable timestamps found — time analysis unavailable."}

    first_ts = parseable["ts"].iloc[0]
    last_ts  = parseable["ts"].iloc[-1]

    # Seconds from the first event for each row
    df["relative_seconds"] = (df["ts"] - first_ts).dt.total_seconds()

    duration_s = (last_ts - first_ts).total_seconds()
    rate_per_min = (len(parseable) / duration_s * 60) if duration_s > 0 else 0.0

    # Events whose timestamp is earlier than the previous event
    out_of_order = int((parseable["ts"].diff().dt.total_seconds().dropna() < 0).sum())

    # Rapid-repeat errors: same error code firing multiple times within threshold
    rapid = _detect_rapid_repeats(parseable)

    return {
        "log_duration_seconds":  round(duration_s, 2),
        "event_rate_per_minute": round(rate_per_min, 2),
        "out_of_order_events":   out_of_order,
        "rapid_repeat_errors":   rapid,
    }


def _detect_rapid_repeats(df: pd.DataFrame) -> list[dict]:
    """
    For each error/critical code that appears more than once, compute the
    average interval between occurrences. Return codes where that average
    falls below RAPID_REPEAT_THRESHOLD_SECONDS — these suggest a tight loop
    or stuck-retry condition rather than isolated failures.
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