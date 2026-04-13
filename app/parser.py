"""
parser.py
Validates incoming JSON log payloads against the expected schema.
Raises ValueError with a clear message if required fields are missing or malformed.
"""

REQUIRED_TOP_LEVEL = {"device_id", "timestamp", "events"}
REQUIRED_EVENT_FIELDS = {"event_id", "timestamp", "level", "message"}
VALID_LEVELS = {"INFO", "WARNING", "ERROR", "CRITICAL"}


def parse_log(data: dict) -> dict:
    """
    Validate and normalize a raw JSON log payload.

    Args:
        data: Raw parsed JSON dict from the uploaded file.

    Returns:
        Validated and normalized log dict.

    Raises:
        ValueError: If required fields are missing or values are invalid.
    """
    # Check top-level required fields
    missing = REQUIRED_TOP_LEVEL - data.keys()
    if missing:
        raise ValueError(f"Missing required top-level fields: {missing}")

    if not isinstance(data["events"], list):
        raise ValueError("'events' must be a list.")

    if len(data["events"]) == 0:
        raise ValueError("'events' list is empty. Nothing to triage.")

    # Validate each event
    validated_events = []
    for i, event in enumerate(data["events"]):
        if not isinstance(event, dict):
            raise ValueError(f"Event at index {i} is not a valid object.")

        missing_event_fields = REQUIRED_EVENT_FIELDS - event.keys()
        if missing_event_fields:
            raise ValueError(f"Event {i} missing fields: {missing_event_fields}")

        level = str(event["level"]).upper()
        if level not in VALID_LEVELS:
            raise ValueError(
                f"Event {i} has invalid level '{event['level']}'. "
                f"Must be one of: {VALID_LEVELS}"
            )

        validated_events.append({
            "event_id": str(event["event_id"]),
            "timestamp": str(event["timestamp"]),
            "level": level,
            "message": str(event["message"]),
            "code": str(event.get("code", "N/A")),
            "component": str(event.get("component", "unknown"))
        })

    return {
        "device_id": str(data["device_id"]),
        "timestamp": str(data["timestamp"]),
        "firmware_version": str(data.get("firmware_version", "unknown")),
        "events": validated_events
    }
