"""
summarizer.py
Hybrid pipeline — the deterministic parser + analyzer feed into an LLM that
generates root_cause and summary using instruction-style decomposed prompts.

Prompt format (must match train.py build_prompt exactly):
  "Analyze the following device log findings.
   Severity: HIGH
   Anomalies:
   - [CRITICAL_EVENT] motor: Motor controller unresponsive
   - [REPEATED_ERROR] calibration: Error code 3001 repeated 3 times
   Provide: (1) root cause in one sentence, (2) 2-3 sentence triage summary."

Expected output format:
  "Root cause: <one sentence>. Summary: <2-3 sentences>."

Model load order: Shriya613/surgical-log-triage → google/flan-t5-base → rule-based fallback
"""

import logging
import re
from functools import lru_cache

logger = logging.getLogger(__name__)

FINETUNED_MODEL = "Shriya613/surgical-log-triage"
BASE_MODEL      = "google/flan-t5-base"


@lru_cache(maxsize=1)
def _load_model():
    """Load tokenizer + model once at startup; reuse across requests."""
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except ImportError:
        logger.error("transformers not installed. Run: pip install transformers torch sentencepiece")
        return None, None

    for model_id in [FINETUNED_MODEL, BASE_MODEL]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model     = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            model.eval()
            logger.info(f"Loaded model: {model_id}")
            return tokenizer, model
        except Exception as e:
            logger.warning(f"Could not load {model_id}: {e}")

    logger.error("All models failed. Falling back to rule-based triage.")
    return None, None


def generate_triage(findings: dict, similar_logs: list[dict] | None = None) -> dict:
    """
    Generate root_cause and summary from structured findings.

    Args:
        findings:     Output dict from analyzer.analyze_log()
        similar_logs: Optional list of similar past log summaries from retriever

    Returns:
        {"root_cause": str, "summary": str}
    """
    tokenizer, model = _load_model()
    if tokenizer is None:
        return _fallback_triage(findings)

    prompt = _build_prompt(findings, similar_logs or [])

    try:
        import torch
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                num_beams=4,
                early_stopping=True,
            )
        raw = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return _parse_output(raw, findings)

    except Exception as e:
        logger.error(f"Model inference error: {e}")
        return _fallback_triage(findings)


def _build_prompt(findings: dict, similar_logs: list[dict]) -> str:
    """
    Instruction-style prompt with decomposed reasoning.
    Caps at 6 anomalies and 2 retrieval examples to stay within 512 tokens.

    This format MUST match build_prompt() in train.py exactly so the model
    sees the same structure at training time and inference time.
    """
    anomalies = findings.get("anomalies", [])
    severity  = findings.get("severity", "low")

    anomaly_lines = [
        f"- [{a['type'].upper()}] {a['component']}: {a['detail']}"
        for a in anomalies[:6]
    ]
    anomaly_text = "\n".join(anomaly_lines) if anomaly_lines else "- No anomalies detected"

    rag_section = ""
    if similar_logs:
        examples = [
            f"  Past case ({l.get('severity', 'unknown')} severity): "
            f"{l.get('summary', '')[:100]}"
            for l in similar_logs[:2]
        ]
        rag_section = "\nSimilar past cases:\n" + "\n".join(examples) + "\n"

    return (
        f"Analyze the following device log findings.\n"
        f"Severity: {severity.upper()}\n"
        f"Anomalies:\n{anomaly_text}\n"
        f"{rag_section}"
        f"Provide: (1) root cause in one sentence, (2) 2-3 sentence triage summary."
    )


def _parse_output(raw: str, findings: dict) -> dict:
    """
    Extract root_cause and summary from model output.
    Expects: "Root cause: ... Summary: ..."
    Falls back to rule-based if parsing fails.
    """
    root_cause = ""
    summary    = ""

    # Try "Root cause: ... Summary: ..." format
    rc_match  = re.search(r"root\s*cause\s*[:\-]\s*(.+?)(?=summary\s*[:\-]|$)",
                          raw, re.IGNORECASE | re.DOTALL)
    sum_match = re.search(r"summary\s*[:\-]\s*(.+)",
                          raw, re.IGNORECASE | re.DOTALL)

    if rc_match:
        root_cause = rc_match.group(1).strip()
    if sum_match:
        summary = sum_match.group(1).strip()

    # Fallback: split on first sentence boundary
    if not root_cause and raw:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw) if s.strip()]
        root_cause = sentences[0] if sentences else raw
        summary    = " ".join(sentences[1:]) if len(sentences) > 1 else raw

    if not root_cause or len(root_cause) < 5:
        return _fallback_triage(findings)

    return {
        "root_cause": root_cause[:300],
        "summary":    (summary or root_cause)[:600],
    }


def _fallback_triage(findings: dict) -> dict:
    """Deterministic rule-based triage used when no model is available."""
    severity  = findings.get("severity", "low")
    anomalies = findings.get("anomalies", [])
    components = findings.get("component_error_breakdown", {})

    critical = [a for a in anomalies if a["type"] == "critical_event"]
    repeated = [a for a in anomalies if a["type"] == "repeated_error"]
    rapid    = [a for a in anomalies if a["type"] == "rapid_repeat"]

    # Root cause: pick the most specific available signal
    if critical:
        root_cause = (
            f"Critical failure in {critical[0]['component']} component: "
            f"{critical[0]['detail']}."
        )
    elif repeated:
        root_cause = (
            f"Repeated error code {repeated[0]['code']} "
            f"({repeated[0]['occurrences']}x) in {repeated[0]['component']} "
            f"indicates a persistent fault."
        )
    elif rapid:
        root_cause = (
            f"Rapid-repeat error in {rapid[0]['component']} "
            f"({rapid[0]['occurrences']}x, avg {rapid[0].get('avg_interval_seconds', '?')}s interval) "
            f"suggests a stuck-retry loop."
        )
    elif anomalies:
        root_cause = f"{anomalies[0]['component']} anomaly: {anomalies[0]['detail']}."
    else:
        root_cause = "No significant root cause identified. Log appears nominal."

    # Summary
    parts = [f"Severity assessed as {severity.upper()}."]
    if critical:
        parts.append(
            f"{len(critical)} CRITICAL event(s) detected — "
            "device must be taken offline immediately."
        )
    if repeated:
        codes = ", ".join(f"{a['code']} ({a['occurrences']}x)" for a in repeated)
        parts.append(f"Repeated error codes: {codes}.")
    if components:
        top = sorted(components.items(), key=lambda x: -x[1])[:2]
        parts.append(
            f"Most affected components: {', '.join(c for c, _ in top)}."
        )
    if not anomalies:
        parts.append("No anomalies detected — device log appears healthy.")

    return {
        "root_cause": root_cause,
        "summary":    " ".join(parts),
    }


# ── Backward-compatible alias ─────────────────────────────────────────────────

def summarize_findings(findings: dict) -> str:
    """Legacy alias — returns only the summary string."""
    return generate_triage(findings).get("summary", "")