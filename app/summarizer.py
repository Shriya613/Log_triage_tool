"""
summarizer.py
Uses a locally loaded HuggingFace model to generate a plain-English triage
summary from structured findings. Tries the fine-tuned model first
(Shriya613/surgical-log-triage), falls back to google/flan-t5-base, then
falls back to a rule-based summary if no model is available.
"""

import json
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

FINETUNED_MODEL = "Shriya613/surgical-log-triage"
BASE_MODEL = "google/flan-t5-base"


@lru_cache(maxsize=1)
def _load_model():
    """
    Load tokenizer and model once at startup, cache for reuse across requests.
    Tries the fine-tuned model first, then the base model.
    Returns (tokenizer, model) or (None, None) if loading fails entirely.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except ImportError:
        logger.error("transformers is not installed. Run: pip install transformers torch sentencepiece")
        return None, None

    for model_id in [FINETUNED_MODEL, BASE_MODEL]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            model.eval()
            logger.info(f"Loaded model: {model_id}")
            return tokenizer, model
        except Exception as e:
            logger.warning(f"Could not load {model_id}: {e}")

    logger.error("All models failed to load. Falling back to rule-based summary.")
    return None, None


def summarize_findings(findings: dict) -> str:
    """
    Generate a plain-English triage summary from structured findings.

    Args:
        findings: The anomaly findings dict from analyzer.py

    Returns:
        A 3-5 sentence triage summary string.
    """
    tokenizer, model = _load_model()

    if tokenizer is None:
        return _fallback_summary(findings)

    prompt = _build_prompt(findings)

    try:
        import torch
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                num_beams=4,
                early_stopping=True,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    except Exception as e:
        logger.error(f"Model inference error: {e}")
        return _fallback_summary(findings)


def _build_prompt(findings: dict) -> str:
    """
    Build a concise instruction prompt from findings.
    Strips bulky flagged_events list to stay within the 512-token input limit.
    """
    compact = {k: v for k, v in findings.items() if k != "flagged_events"}
    return (
        "Summarize the following medical device log findings in plain English. "
        "State what failed, which components are affected, the risk level, "
        "and whether immediate action is required.\n\n"
        f"Findings:\n{json.dumps(compact, indent=2)}"
    )


def _fallback_summary(findings: dict) -> str:
    """Rule-based summary used when no model is available."""
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