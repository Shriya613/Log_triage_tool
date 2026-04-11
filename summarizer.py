"""
summarizer.py
Takes structured findings from analyzer.py and uses an LLM
to generate a plain-English triage summary for engineers.
Falls back gracefully if the API key is missing or call fails.
"""

import os
import json
import logging

import openai

logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")


def summarize_findings(findings: dict) -> str:
    """
    Send structured findings to an LLM and return a plain-English triage summary.

    Args:
        findings: The anomaly findings dict from analyzer.py

    Returns:
        A 3-5 sentence triage summary string.
    """
    if not openai.api_key:
        logger.warning("OPENAI_API_KEY not set. Returning rule-based summary.")
        return _fallback_summary(findings)

    prompt = _build_prompt(findings)

    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical device software engineer reviewing device logs. "
                        "Write a concise, factual triage summary (3-5 sentences) for an engineer. "
                        "Be specific about what failed, which components are affected, "
                        "and whether immediate action is needed. Do not speculate beyond the data."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()

    except openai.OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
        return _fallback_summary(findings)


def _build_prompt(findings: dict) -> str:
    """Build a compact prompt from findings dict."""
    return (
        f"Device log analysis findings:\n"
        f"{json.dumps(findings, indent=2)}\n\n"
        f"Summarize: what happened, which components are affected, "
        f"risk level, and recommended next action."
    )


def _fallback_summary(findings: dict) -> str:
    """
    Rule-based summary when LLM is unavailable.
    Ensures the tool still works without an API key.
    """
    risk = findings.get("risk_level", "UNKNOWN")
    critical = len(findings.get("critical_events", []))
    flagged = findings.get("total_flagged_events", 0)
    repeated = findings.get("repeated_error_codes", {})
    components = findings.get("component_error_breakdown", {})

    parts = [f"Risk level assessed as {risk}."]

    if critical > 0:
        parts.append(f"{critical} CRITICAL event(s) detected — immediate review recommended.")
    if flagged > 0:
        parts.append(f"{flagged} total error/critical event(s) flagged.")
    if repeated:
        codes = ", ".join(f"{k} ({v}x)" for k, v in repeated.items())
        parts.append(f"Repeated error codes detected: {codes}.")
    if components:
        top = sorted(components.items(), key=lambda x: -x[1])[:3]
        comp_str = ", ".join(f"{c} ({n} errors)" for c, n in top)
        parts.append(f"Most affected components: {comp_str}.")
    if risk == "NOMINAL":
        parts.append("No significant anomalies detected. Log appears healthy.")

    return " ".join(parts)
