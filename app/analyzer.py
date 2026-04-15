"""
analyzer.py
Deterministic rule-based anomaly detection.
Produces a typed anomalies list and a three-level severity score.
No LLM involved — this is the tool layer that feeds the LLM in summarizer.py.

Anomaly types:
  critical_event   — CRITICAL level event
  repeated_error   — same error code fires 3+ times
  warning_cluster  — warning count exceeds LOW threshold
  stuck_state      — same message repeats 3+ times
  rapid_repeat     — error fires with avg interval < threshold
  out_of_order     — event timestamp earlier than its predecessor
  extreme_value    — numeric sensor value exceeds ±EXTREME_VALUE_THRESHOLD
  missing_field    — required field absent (surfaced from schema_issues)
  escalation       — warn → error → critical progression in one session
"""

from collections import Counter
from typing import Any

EXTREME_VALUE_THRESHOLD = 1000.0
WARNING_CLUSTER_THRESHOLD = 5        # ≥N warnings → warning_cluster anomaly
REPEATED_CODE_THRESHOLD = 3          # same error code ≥N times → repeated_error


def analyze_log(parsed: dict) -> dict[str, Any]:
    """
    Run rule-based analysis on a parsed log payload.

    Args:
        parsed: Output dict from parser.parse_log()

    Returns:
        anomalies (list[dict])          — typed anomaly objects
        severity (str)                  — "low" | "medium" | "high"
        level_counts (dict)
        component_error_breakdown (dict)
        flagged_events (list)           — kept for summarizer compatibility
    """
    events       = parsed.get("events", [])
    time_analysis = parsed.get("time_analysis", {})
    schema_issues = parsed.get("schema_issues", [])
    anomalies: list[dict] = []

    level_counts = Counter(e["level"] for e in events)
    flagged      = [e for e in events if e["level"] in {"ERROR", "CRITICAL"}]

    # ── Critical events ────────────────────────────────────────────────────────
    for e in events:
        if e["level"] == "CRITICAL":
            anomalies.append({
                "type":        "critical_event",
                "component":   e["component"],
                "detail":      e["message"],
                "code":        e["code"] if e["code"] != "N/A" else None,
                "occurrences": 1,
            })

    # ── Repeated error codes ───────────────────────────────────────────────────
    code_counts = Counter(
        e["code"] for e in events
        if e["level"] in {"ERROR", "CRITICAL"} and e["code"] != "N/A"
    )
    for code, count in code_counts.items():
        if count >= REPEATED_CODE_THRESHOLD:
            component = next(
                (e["component"] for e in events if e["code"] == code), "unknown"
            )
            anomalies.append({
                "type":        "repeated_error",
                "component":   component,
                "detail":      f"Error code {code} repeated {count} times",
                "code":        code,
                "occurrences": count,
            })

    # ── Warning cluster ────────────────────────────────────────────────────────
    warn_count = level_counts.get("WARNING", 0)
    if warn_count >= WARNING_CLUSTER_THRESHOLD:
        # Find the most common warning component
        warn_components = Counter(
            e["component"] for e in events if e["level"] == "WARNING"
        )
        top_comp = warn_components.most_common(1)[0][0] if warn_components else "unknown"
        anomalies.append({
            "type":        "warning_cluster",
            "component":   top_comp,
            "detail":      f"{warn_count} WARNING events exceed the LOW-risk threshold",
            "code":        None,
            "occurrences": warn_count,
        })

    # ── Stuck state (repeated messages) ───────────────────────────────────────
    msg_counts = Counter(e["message"] for e in events)
    for msg, count in msg_counts.items():
        if count < 3:
            continue
        component = next(
            (e["component"] for e in events if e["message"] == msg), "unknown"
        )
        # Avoid duplicating what repeated_error already caught
        already_flagged = any(
            a["type"] == "repeated_error" and a["component"] == component
            for a in anomalies
        )
        if not already_flagged:
            anomalies.append({
                "type":        "stuck_state",
                "component":   component,
                "detail":      f"Message repeated {count}x: '{msg[:80]}'",
                "code":        None,
                "occurrences": count,
            })

    # ── Rapid-repeat errors (from time analysis) ───────────────────────────────
    for rr in time_analysis.get("rapid_repeat_errors", []):
        anomalies.append({
            "type":        "rapid_repeat",
            "component":   rr["component"],
            "detail":      (
                f"Code {rr['code']} fired {rr['occurrences']}x "
                f"(avg interval {rr['avg_interval_seconds']}s < "
                f"{10}s threshold)"
            ),
            "code":        rr["code"],
            "occurrences": rr["occurrences"],
        })

    # ── Out-of-order timestamps ────────────────────────────────────────────────
    oot = time_analysis.get("out_of_order_events", 0)
    if oot > 0:
        anomalies.append({
            "type":        "out_of_order",
            "component":   "system",
            "detail":      f"{oot} event(s) have timestamps earlier than the preceding event",
            "code":        None,
            "occurrences": oot,
        })

    # ── Extreme sensor values ──────────────────────────────────────────────────
    for e in events:
        if e.get("value") is None:
            continue
        try:
            v = float(e["value"])
            if abs(v) > EXTREME_VALUE_THRESHOLD:
                anomalies.append({
                    "type":        "extreme_value",
                    "component":   e["component"],
                    "detail":      (
                        f"Sensor reading {v} exceeds threshold "
                        f"±{EXTREME_VALUE_THRESHOLD}"
                    ),
                    "code":        e["code"] if e["code"] != "N/A" else None,
                    "occurrences": 1,
                })
        except (TypeError, ValueError):
            pass

    # ── Schema issues as missing_field anomalies ───────────────────────────────
    for issue in schema_issues:
        if "missing" in issue.lower():
            anomalies.append({
                "type":        "missing_field",
                "component":   "schema",
                "detail":      issue,
                "code":        None,
                "occurrences": 1,
            })

    # ── Escalation pattern: warn → error → critical in sequence ───────────────
    has_warn     = level_counts.get("WARNING", 0) >= 2
    has_error    = level_counts.get("ERROR", 0) >= 1
    has_critical = level_counts.get("CRITICAL", 0) >= 1
    if has_warn and has_error and has_critical:
        # Only add escalation if there is also a critical_event anomaly already
        if any(a["type"] == "critical_event" for a in anomalies):
            comp = next(
                (a["component"] for a in anomalies if a["type"] == "critical_event"),
                "unknown",
            )
            anomalies.append({
                "type":        "escalation",
                "component":   comp,
                "detail":      (
                    f"{level_counts['WARNING']} warnings → "
                    f"{level_counts['ERROR']} errors → "
                    f"{level_counts['CRITICAL']} critical: progressive failure pattern"
                ),
                "code":        None,
                "occurrences": sum(level_counts[k] for k in ("WARNING", "ERROR", "CRITICAL")),
            })

    # ── Severity ───────────────────────────────────────────────────────────────
    critical_count  = level_counts.get("CRITICAL", 0)
    error_count     = level_counts.get("ERROR", 0)
    has_critical_a  = any(a["type"] in {"critical_event", "rapid_repeat", "escalation"} for a in anomalies)
    has_repeated    = any(a["type"] == "repeated_error" for a in anomalies)

    if critical_count > 0 or has_critical_a or has_repeated:
        severity = "high"
    elif error_count >= 2 or len(anomalies) >= 3:
        severity = "medium"
    else:
        severity = "low"

    return {
        "anomalies":                anomalies,
        "severity":                 severity,
        "level_counts":             dict(level_counts),
        "component_error_breakdown": dict(Counter(
            e["component"] for e in events if e["level"] in {"ERROR", "CRITICAL"}
        )),
        "flagged_events": flagged,
    }