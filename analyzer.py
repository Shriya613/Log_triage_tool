"""
analyzer.py
Analyzes validated log events to detect anomalies, error patterns,
repeated failures, and suspicious sequences.
Returns structured findings — no LLM involved here.
"""

from collections import Counter
from typing import Any


def analyze_log(parsed: dict) -> dict[str, Any]:
    """
    Run rule-based analysis on a validated log payload.

    Args:
        parsed: Validated log dict from parser.py

    Returns:
        findings dict with error counts, anomalies, repeated codes, and flagged events.
    """
    events = parsed["events"]

    # --- Level counts ---
    level_counts = Counter(e["level"] for e in events)

    # --- Flag high-severity events ---
    flagged = [
        e for e in events
        if e["level"] in {"ERROR", "CRITICAL"}
    ]

    # --- Detect repeated error codes (same code appearing 3+ times) ---
    code_counts = Counter(
        e["code"] for e in events
        if e["level"] in {"ERROR", "CRITICAL"} and e["code"] != "N/A"
    )
    repeated_codes = {
        code: count
        for code, count in code_counts.items()
        if count >= 3
    }

    # --- Detect repeated messages (possible loop / stuck state) ---
    message_counts = Counter(e["message"] for e in events)
    repeated_messages = {
        msg: count
        for msg, count in message_counts.items()
        if count >= 3
    }

    # --- Detect CRITICAL events ---
    critical_events = [e for e in events if e["level"] == "CRITICAL"]

    # --- Component breakdown ---
    component_error_counts = Counter(
        e["component"]
        for e in events
        if e["level"] in {"ERROR", "CRITICAL"}
    )

    # --- Overall risk level ---
    if critical_events or repeated_codes:
        risk_level = "HIGH"
    elif level_counts.get("ERROR", 0) >= 3:
        risk_level = "MEDIUM"
    elif level_counts.get("WARNING", 0) >= 5:
        risk_level = "LOW"
    else:
        risk_level = "NOMINAL"

    return {
        "risk_level": risk_level,
        "level_counts": dict(level_counts),
        "total_flagged_events": len(flagged),
        "critical_events": critical_events,
        "repeated_error_codes": repeated_codes,
        "repeated_messages": repeated_messages,
        "component_error_breakdown": dict(component_error_counts),
        "flagged_events": flagged
    }
