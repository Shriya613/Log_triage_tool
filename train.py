"""
train.py
Fine-tunes google/flan-t5-base on surgical device log triage data using LoRA.

Key design choices vs the previous version:
  - LoRA (r=16, target q+v projections) — trains ~3% of parameters
    vs full fine-tuning, less GPU memory, better generalisation on small data.
  - Instruction-style prompts that match app/summarizer._build_prompt() exactly.
  - Training target format: "Root cause: <…>. Summary: <…>."
  - 30% of examples get noise injected (missing fields, extreme values,
    out-of-order events) so the model learns to handle messy real-world input.
  - Updated generators produce structured anomalies list + root_cause field
    matching the new analyzer.py output schema.

Data sources:
  1. Synthetic generators — 500 examples, 7 scenario types, 5-8 varied
     root_cause + summary templates per type.
  2. External CSV (optional) — AI4I 2020 Predictive Maintenance dataset from
     Kaggle (search "AI4I 2020 predictive maintenance"). Pass the path with
     --extra-data path/to/ai4i2020.csv.

Colab workflow:
    !python train.py                                    # synthetic only
    !python train.py --extra-data /content/ai4i.csv    # + AI4I data
    !python train.py --push --hub-token YOUR_TOKEN      # + publish
"""

import argparse
import copy
import json
import random
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

# ── Constants ──────────────────────────────────────────────────────────────────

BASE_MODEL  = "google/flan-t5-base"
OUTPUT_DIR  = "model_output"
HF_REPO     = "Shriya613/surgical-log-triage"

COMPONENTS = [
    "motor", "calibration", "sensor", "power", "navigation",
    "safety", "core", "actuator", "valve", "firmware", "comms", "battery",
]
ERROR_CODES = {
    "motor":       "5001",
    "calibration": "3001",
    "sensor":      "2001",
    "power":       "9001",
    "navigation":  "4001",
    "safety":      "5002",
    "core":        "1001",
    "actuator":    "6001",
    "valve":       "6002",
    "firmware":    "7001",
    "comms":       "8001",
    "battery":     "9002",
}

# ── Anomaly helpers ────────────────────────────────────────────────────────────

def _anom(type_, component, detail, code=None, occurrences=None):
    return {"type": type_, "component": component, "detail": detail,
            "code": code, "occurrences": occurrences}


# ── Synthetic data generators ──────────────────────────────────────────────────
# Each returns: {findings: {anomalies, severity, …}, root_cause: str, summary: str}
# root_cause + summary both have 5-8 varied templates — forces the model to learn
# the reasoning pattern rather than memorising a single phrasing.

def _make_nominal():
    info_count = random.randint(5, 30)
    root_causes = [
        "No faults detected — all subsystems completed their cycle within normal parameters.",
        "Zero anomalies across all components; no root cause investigation required.",
        f"The device logged {info_count} informational events with no deviations from baseline.",
        "All diagnostic signals are within expected ranges. No fault condition identified.",
        "System operated nominally throughout the session with no error or warning events.",
    ]
    summaries = [
        f"Severity LOW: {info_count} INFO events, zero flagged issues. "
        "No component shows signs of stress or failure. Device is cleared for continued operation.",

        "Log analysis complete. Zero anomalies detected across all subsystems. "
        "Event distribution is entirely informational. No maintenance action is needed.",

        f"Clean run: {info_count} INFO events, no warnings or errors. "
        "Risk is LOW. Proceed with normal operations.",

        "All logged events are at INFO level with no anomalies across any subsystem. "
        "No intervention required.",

        f"No issues found. The device completed its cycle with {info_count} informational "
        "events only. No follow-up required.",

        "Device log is fully nominal. No error codes, no warnings, no critical events. "
        "Cleared for next operational cycle.",
    ]
    return {
        "findings": {
            "anomalies":                [],
            "severity":                 "low",
            "level_counts":             {"INFO": info_count},
            "component_error_breakdown": {},
        },
        "root_cause": random.choice(root_causes),
        "summary":    random.choice(summaries),
    }


def _make_low():
    comp       = random.choice(COMPONENTS)
    warn_count = random.randint(5, 15)
    info_count = random.randint(3, 12)
    root_causes = [
        f"Elevated warning activity in the {comp} subsystem indicates early-stage drift or wear.",
        f"{warn_count} warnings from {comp} suggest the component is operating near its tolerance limit.",
        f"The {comp} subsystem is generating above-baseline warnings, pointing to developing degradation.",
        f"Warning cluster in {comp}: likely a minor calibration offset or sensor drift, not yet critical.",
        f"Repeated warnings in {comp} may indicate intermittent contact, loose connection, or parameter drift.",
    ]
    summaries = [
        f"Severity LOW: {warn_count} WARNING events in the {comp} subsystem. "
        "No errors or critical failures. Review at the next scheduled maintenance window.",

        f"{warn_count} warnings recorded in {comp} — below error threshold but above normal. "
        f"Risk is LOW. Monitor the {comp} subsystem; schedule a diagnostic if pattern persists.",

        f"Elevated warning activity in {comp} ({warn_count} events). No immediate action required, "
        "but continued operation should be monitored closely.",

        f"The {comp} subsystem generated {warn_count} WARNING events. Risk: LOW. "
        f"Device is operational — review {comp} at the earliest opportunity.",

        f"Risk LOW: {warn_count} warnings in {comp}, {info_count} informational events elsewhere. "
        f"Flag {comp} for non-urgent inspection.",

        f"Warnings in {comp} ({warn_count}) have crossed the LOW-risk threshold. "
        "Recommend a non-urgent diagnostic to rule out progressive fault.",
    ]
    return {
        "findings": {
            "anomalies": [
                _anom("warning_cluster", comp,
                      f"{warn_count} WARNING events exceed the LOW-risk threshold",
                      occurrences=warn_count),
            ],
            "severity":                 "low",
            "level_counts":             {"INFO": info_count, "WARNING": warn_count},
            "component_error_breakdown": {},
        },
        "root_cause": random.choice(root_causes),
        "summary":    random.choice(summaries),
    }


def _make_medium():
    comp  = random.choice(COMPONENTS)
    code  = ERROR_CODES[comp]
    count = random.randint(2, 4)   # below repeated_error threshold of 3 — or at 3
    info  = random.randint(2, 10)
    root_causes = [
        f"Error code {code} in {comp} indicates a recurring fault that has not yet reached critical severity.",
        f"Multiple {comp} errors (code {code}) suggest a developing hardware or firmware fault.",
        f"The {comp} subsystem is reporting {count} error events — consistent with a partial failure or marginal condition.",
        f"Error code {code} has appeared {count} times in {comp}, indicating an intermittent fault.",
        f"The {comp} component exceeded its error threshold with {count} occurrences of code {code}.",
    ]
    summaries = [
        f"Severity MEDIUM: {count} ERROR events in the {comp} subsystem (code {code}). "
        "No critical failures, but the error count exceeds threshold. "
        f"Diagnostic review of {comp} recommended before next use.",

        f"{count} errors flagged in {comp} (code {code}). Risk is MEDIUM — "
        f"device is operable but degraded. Investigate {comp} before next deployment.",

        f"Error code {code} appeared {count} times in {comp}, pushing risk to MEDIUM. "
        "No critical events, but recurring errors suggest a developing fault. "
        "Schedule a targeted inspection.",

        f"MEDIUM risk: {count} errors from {comp} (code {code}). "
        "This is above the warning threshold but below critical. "
        f"Recommend a diagnostic sweep of the {comp} module.",

        f"Log shows {count} {comp} errors (code {code}) alongside {info} informational events. "
        f"Risk: MEDIUM. {comp} should be checked at the earliest opportunity.",

        f"The {comp} subsystem reported {count} ERROR-level events (code {code}). "
        "Risk: MEDIUM. Device completed its cycle but requires review before next deployment.",
    ]
    return {
        "findings": {
            "anomalies": [
                _anom("repeated_error", comp,
                      f"Error code {code} repeated {count} times",
                      code=code, occurrences=count),
            ] if count >= 3 else [
                _anom("stuck_state", comp,
                      f"Error code {code} appeared {count} times",
                      code=code, occurrences=count),
            ],
            "severity":                 "medium",
            "level_counts":             {"INFO": info, "ERROR": count},
            "component_error_breakdown": {comp: count},
        },
        "root_cause": random.choice(root_causes),
        "summary":    random.choice(summaries),
    }


def _make_high_repeated():
    comp  = random.choice(COMPONENTS)
    code  = ERROR_CODES[comp]
    count = random.randint(3, 8)
    info  = random.randint(1, 6)
    root_causes = [
        f"Repeated firing of error code {code} in {comp} ({count}x) indicates a persistent fault, not a transient event.",
        f"Error code {code} has repeated {count} times in {comp} — consistent with a stuck-retry loop or unresolved hardware fault.",
        f"The {comp} subsystem is locked in an error loop (code {code}, {count} occurrences), pointing to an uncleared fault condition.",
        f"Persistent error {code} in {comp} ({count}x) suggests the root issue is not self-resolving and requires engineer intervention.",
        f"Code {code} repeated {count} times in {comp}: likely cause is an unresolved hardware fault or firmware regression.",
    ]
    summaries = [
        f"Severity HIGH: error code {code} fired {count} times in {comp}. "
        "Repeated identical codes indicate a persistent fault. "
        f"Take {comp} offline and do not clear the log until root cause is confirmed.",

        f"HIGH risk — {comp} is in a repeated error loop (code {code}, {count} occurrences). "
        f"Isolate the {comp} component immediately.",

        f"Error code {code} has repeated {count} times in {comp} — "
        "a clear indicator of a persistent failure mode. "
        f"Risk is HIGH. The {comp} subsystem must be inspected before the device is returned to service.",

        f"The {comp} subsystem is generating the same error code ({code}) "
        f"repeatedly ({count} times). Risk: HIGH. "
        "Escalate to engineering for root cause analysis.",

        f"HIGH risk: {count} instances of error {code} from {comp}. "
        "Repeated firing of one error code is a strong signal of a systematic failure. "
        "Do not resume operation until the fault is diagnosed and cleared.",

        f"Log analysis flagged {count} occurrences of error code {code} from {comp} (risk: HIGH). "
        "Remove device from service and inspect the {comp} module.",
    ]
    return {
        "findings": {
            "anomalies": [
                _anom("repeated_error", comp,
                      f"Error code {code} repeated {count} times",
                      code=code, occurrences=count),
            ],
            "severity":                 "high",
            "level_counts":             {"INFO": info, "ERROR": count},
            "component_error_breakdown": {comp: count},
        },
        "root_cause": random.choice(root_causes),
        "summary":    random.choice(summaries),
    }


def _make_high_critical():
    comp  = random.choice(COMPONENTS)
    code  = ERROR_CODES[comp]
    count = random.randint(1, 3)
    info  = random.randint(1, 5)
    root_causes = [
        f"Critical failure in the {comp} subsystem (code {code}) — hard fault requiring immediate offline action.",
        f"The {comp} component raised {count} CRITICAL alert(s) (code {code}), indicating it has exceeded safe operating parameters.",
        f"CRITICAL event in {comp} (code {code}): the component has entered a failure state that cannot self-recover in the field.",
        f"Hard failure confirmed: {comp} subsystem code {code} — this exceeds the recoverable fault threshold.",
        f"The {comp} subsystem triggered a CRITICAL-level fault (code {code}), indicating a severe hardware or safety condition.",
    ]
    summaries = [
        f"Severity HIGH: {count} CRITICAL event(s) in the {comp} subsystem (code {code}). "
        "CRITICAL events indicate a hard failure — device must be taken offline immediately.",

        f"CRITICAL failure in {comp} (code {code}, {count} event(s)). "
        "Risk is HIGH. This is a hard stop — remove the device from service and inspect.",

        f"{count} CRITICAL-level event(s) detected in {comp} (code {code}). "
        "Immediate action required: power down, quarantine the log, and notify the engineering team.",

        f"Hard failure confirmed: {comp} raised {count} CRITICAL alert(s) (code {code}). "
        "Risk: HIGH. Do not attempt a soft reset without engineering sign-off.",

        f"Risk HIGH — {comp} experienced {count} CRITICAL failure(s) (code {code}). "
        "Device has exceeded safe operating parameters and must be reviewed before next use.",

        f"The {comp} subsystem has entered a CRITICAL failure state ({count} event(s), code {code}). "
        "Cease operation, preserve the log, and engage the support team for a full failure analysis.",
    ]
    return {
        "findings": {
            "anomalies": [
                _anom("critical_event", comp,
                      f"{comp.capitalize()} subsystem failure",
                      code=code, occurrences=count),
            ],
            "severity":                 "high",
            "level_counts":             {"INFO": info, "CRITICAL": count},
            "component_error_breakdown": {comp: count},
        },
        "root_cause": random.choice(root_causes),
        "summary":    random.choice(summaries),
    }


def _make_high_multi():
    comp_a, comp_b = random.sample(COMPONENTS, 2)
    code_a, code_b = ERROR_CODES[comp_a], ERROR_CODES[comp_b]
    err_count = random.randint(3, 6)
    root_causes = [
        f"Simultaneous CRITICAL failure in {comp_a} and repeated errors in {comp_b} suggest a shared root cause — possibly a power rail or communication bus fault.",
        f"Multi-component failure across {comp_a} and {comp_b}: concurrent failures rarely arise independently; investigate shared infrastructure first.",
        f"The {comp_a} CRITICAL event and {comp_b} repeated error loop indicate a cascading fault — one subsystem likely triggered the other.",
        f"Two subsystems failing simultaneously ({comp_a} and {comp_b}) points to a systemic issue rather than isolated component wear.",
        f"Concurrent failures in {comp_a} (CRITICAL) and {comp_b} (error loop) are consistent with a bus fault, power anomaly, or firmware regression.",
    ]
    summaries = [
        f"Severity HIGH: CRITICAL failure in {comp_a} (code {code_a}) and "
        f"repeated error {code_b} in {comp_b} ({err_count}x). "
        "Multi-component failure suggests a shared root cause. "
        f"Isolate both {comp_a} and {comp_b} and run a full system diagnostic.",

        f"Two subsystems affected — {comp_a} (CRITICAL, code {code_a}) and "
        f"{comp_b} (repeated error {code_b}, {err_count} times). Risk: HIGH. "
        "Simultaneous failures across components are rarely independent.",

        f"HIGH risk: cascading failure across {comp_a} and {comp_b}. "
        f"{comp_a} raised a CRITICAL event (code {code_a}); "
        f"{comp_b} logged {err_count} repeated errors (code {code_b}). "
        "Take the full device offline.",

        f"Multi-component fault: {comp_a} CRITICAL (code {code_a}), "
        f"{comp_b}: {err_count} repeated errors (code {code_b}). Risk: HIGH. "
        "Escalate to systems engineering immediately.",

        f"Risk HIGH — {comp_a} critical failure and {comp_b} repeated error state ({err_count}x). "
        "Assume the failure is related until proven otherwise. Full device quarantine required.",
    ]
    return {
        "findings": {
            "anomalies": [
                _anom("critical_event", comp_a,
                      f"{comp_a.capitalize()} failure", code=code_a, occurrences=1),
                _anom("repeated_error", comp_b,
                      f"Error code {code_b} repeated {err_count} times",
                      code=code_b, occurrences=err_count),
            ],
            "severity":                 "high",
            "level_counts":             {"INFO": random.randint(1, 4),
                                         "ERROR": err_count, "CRITICAL": 1},
            "component_error_breakdown": {comp_a: 1, comp_b: err_count},
        },
        "root_cause": random.choice(root_causes),
        "summary":    random.choice(summaries),
    }


def _make_high_escalating():
    comp       = random.choice(COMPONENTS)
    code       = ERROR_CODES[comp]
    warn_count = random.randint(3, 8)
    err_count  = random.randint(2, 5)
    root_causes = [
        f"Progressive degradation in {comp}: the escalation from warnings to errors to a CRITICAL event indicates a developing hardware fault that reached failure during this session.",
        f"The {comp} subsystem degraded progressively this session (warnings → errors → CRITICAL), consistent with a wearing or overheating component.",
        f"Escalation pattern in {comp} (code {code}): warnings preceded errors, which preceded a CRITICAL event — classic signature of a failing component.",
        f"The {comp} component followed an escalation trajectory through all severity levels, indicating a continuous fault that was not caught early enough.",
        f"Warn-error-critical escalation in {comp} (code {code}) confirms a progressive hardware failure — not a one-time fault.",
    ]
    summaries = [
        f"Severity HIGH: {comp} shows a clear escalation pattern — "
        f"{warn_count} warnings → {err_count} errors → CRITICAL. "
        "This trajectory indicates progressive failure. Device must be taken offline.",

        f"HIGH risk — escalating failure in {comp} (code {code}): "
        f"{warn_count}W → {err_count}E → CRITICAL. "
        "Progressive degradation is a strong predictor of imminent hardware failure.",

        f"The {comp} component degraded progressively: "
        f"{warn_count} warnings, {err_count} errors, then a CRITICAL event (code {code}). "
        "Risk: HIGH. Replace or service before next use.",

        f"Log shows escalating fault in {comp}: warnings ({warn_count}) → "
        f"errors ({err_count}) → CRITICAL. "
        "When a component moves through all severity levels in one session, it is failing. "
        "Immediate offline and inspection required.",

        f"Escalation in {comp}: {warn_count}x WARNING → {err_count}x ERROR → "
        f"1x CRITICAL (code {code}). Risk: HIGH. "
        "Schedule replacement before the next operational cycle.",
    ]
    return {
        "findings": {
            "anomalies": [
                _anom("warning_cluster", comp,
                      f"{warn_count} WARNING events", occurrences=warn_count),
                _anom("critical_event", comp,
                      f"{comp.capitalize()} progressive failure",
                      code=code, occurrences=1),
                _anom("escalation", comp,
                      f"{warn_count} warnings → {err_count} errors → 1 critical: progressive failure pattern",
                      occurrences=warn_count + err_count + 1),
            ],
            "severity":                 "high",
            "level_counts":             {"INFO": random.randint(2, 6),
                                         "WARNING": warn_count,
                                         "ERROR":   err_count,
                                         "CRITICAL": 1},
            "component_error_breakdown": {comp: err_count + 1},
        },
        "root_cause": random.choice(root_causes),
        "summary":    random.choice(summaries),
    }


# (generator_function, n_examples) — total: 500
GENERATORS = [
    (_make_nominal,          60),
    (_make_low,              60),
    (_make_medium,           80),
    (_make_high_repeated,    90),
    (_make_high_critical,    90),
    (_make_high_multi,       60),
    (_make_high_escalating,  60),
]


def generate_synthetic(seed: int = 42) -> list[dict]:
    """Build and shuffle all 500 synthetic examples."""
    random.seed(seed)
    examples = []
    for fn, count in GENERATORS:
        for _ in range(count):
            examples.append(fn())
    random.shuffle(examples)
    return examples


# ── Noise injection ────────────────────────────────────────────────────────────

def _inject_noise(example: dict) -> dict:
    """
    Randomly inject realistic noise into a training example's findings.
    Applied to ~30% of examples so the model learns to reason about messy input.
    Each noise type has its own independent probability.
    """
    ex = copy.deepcopy(example)
    findings = ex["findings"]
    findings.setdefault("anomalies", [])
    findings.setdefault("schema_issues", [])

    # Missing field in one event (~25% chance)
    if random.random() < 0.25:
        idx = random.randint(0, 5)
        field = random.choice(["component", "code", "timestamp"])
        issue = f"Event[{idx}]: missing required field '{field}'"
        findings["schema_issues"].append(issue)
        findings["anomalies"].append({
            "type":        "missing_field",
            "component":   "schema",
            "detail":      issue,
            "code":        None,
            "occurrences": 1,
        })

    # Extreme sensor value (~15% chance)
    if random.random() < 0.15:
        comp = random.choice(COMPONENTS)
        val  = random.randint(1001, 9999)
        findings["anomalies"].append({
            "type":        "extreme_value",
            "component":   comp,
            "detail":      f"Sensor reading {val} exceeds threshold ±1000",
            "code":        None,
            "occurrences": 1,
        })

    # Out-of-order timestamps (~10% chance)
    if random.random() < 0.10:
        n = random.randint(1, 3)
        findings["anomalies"].append({
            "type":        "out_of_order",
            "component":   "system",
            "detail":      f"{n} event(s) have timestamps earlier than the preceding event",
            "code":        None,
            "occurrences": n,
        })

    return ex


# ── External dataset (AI4I 2020 Predictive Maintenance) ───────────────────────

_AI4I_COMPONENT_MAP = {
    "TWF": ("motor",       "5001"),   # Tool Wear Failure
    "HDF": ("sensor",      "2001"),   # Heat Dissipation Failure
    "PWF": ("power",       "9001"),   # Power Failure
    "OSF": ("calibration", "3001"),   # Overstrain Failure
    "RNF": ("core",        "1001"),   # Random Failure
}


def load_ai4i_data(csv_path: str) -> list[dict]:
    """
    Convert AI4I 2020 Predictive Maintenance rows to training examples.
    Rows with no machine failure → NOMINAL examples.
    Single failure flag → HIGH critical example, component patched to the real one.
    Multiple failure flags → HIGH multi-component example.
    """
    try:
        import pandas as pd
    except ImportError:
        print("pandas required for external data loading — skipping.")
        return []

    path = Path(csv_path)
    if not path.exists():
        print(f"\nERROR: File not found: {csv_path}")
        print("To find where your file is in Colab, run:")
        print("  !find / -name '*.csv' 2>/dev/null | grep -i ai4i")
        print("Then re-run with the correct path, e.g.:")
        print("  !python train.py --extra-data /content/ai4i2020.csv\n")
        return []

    df           = pd.read_csv(path)
    failure_cols = [c for c in ["TWF", "HDF", "PWF", "OSF", "RNF"] if c in df.columns]
    if not failure_cols:
        print(f"WARNING: expected failure columns not found in {csv_path} — skipping.")
        return []

    examples = []
    for _, row in df.iterrows():
        active = [c for c in failure_cols if int(row.get(c, 0)) == 1]

        if not active:
            ex = _make_nominal()
        elif len(active) == 1:
            comp, code = _AI4I_COMPONENT_MAP[active[0]]
            ex = _make_high_critical()
            # Patch anomaly to reflect the actual AI4I failure component
            ex["findings"]["anomalies"] = [
                _anom("critical_event", comp,
                      f"{comp.capitalize()} failure ({active[0]})",
                      code=code, occurrences=1)
            ]
            ex["findings"]["component_error_breakdown"] = {comp: 1}
        else:
            ex = _make_high_multi()

        examples.append(ex)

    print(f"  Loaded {len(examples)} examples from {csv_path}")
    return examples


# ── Prompt builder (MUST match app/summarizer._build_prompt exactly) ───────────

def build_prompt(findings: dict) -> str:
    """
    Instruction-style prompt identical to what app/summarizer.py sends at
    inference time. Training and inference prompts must match exactly.
    """
    anomalies = findings.get("anomalies", [])
    severity  = findings.get("severity", "low")

    anomaly_lines = [
        f"- [{a['type'].upper()}] {a['component']}: {a['detail']}"
        for a in anomalies[:6]
    ]
    anomaly_text = "\n".join(anomaly_lines) if anomaly_lines else "- No anomalies detected"

    return (
        f"Analyze the following device log findings.\n"
        f"Severity: {severity.upper()}\n"
        f"Anomalies:\n{anomaly_text}\n"
        f"Provide: (1) root cause in one sentence, (2) 2-3 sentence triage summary."
    )


def build_target(root_cause: str, summary: str) -> str:
    """Target output format — parsed back by app/summarizer._parse_output()."""
    return f"Root cause: {root_cause} Summary: {summary}"


# ── Tokenisation ───────────────────────────────────────────────────────────────

def tokenize(batch, tokenizer):
    inputs = tokenizer(
        batch["input_text"],
        max_length=512,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        text_target=batch["output_text"],
        max_length=128,
        truncation=True,
        padding="max_length",
    )
    # Replace pad token IDs in labels with -100 so the loss ignores padding
    labels["input_ids"] = [
        [(t if t != tokenizer.pad_token_id else -100) for t in lbl]
        for lbl in labels["input_ids"]
    ]
    inputs["labels"] = labels["input_ids"]
    return inputs


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune flan-t5-base with LoRA for log triage")
    parser.add_argument("--push",       action="store_true",
                        help="Push fine-tuned model to HuggingFace Hub")
    parser.add_argument("--hub-token",  type=str, default=None,
                        help="HuggingFace write token (or set HF_TOKEN env var)")
    parser.add_argument("--epochs",     type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lora-rank",  type=int, default=16,
                        help="LoRA rank r (default 16; increase for more capacity)")
    parser.add_argument("--extra-data", type=str, default=None, metavar="CSV_PATH",
                        help="Path to AI4I 2020 CSV for additional training examples")
    args = parser.parse_args()

    # ── Step 1: build dataset ─────────────────────────────────────────────────
    print("Generating synthetic training data...")
    all_examples = generate_synthetic()
    print(f"  Synthetic: {len(all_examples)} examples")

    if args.extra_data:
        print(f"Loading external data from {args.extra_data}...")
        external = load_ai4i_data(args.extra_data)
        all_examples.extend(external)
        random.shuffle(all_examples)

    print("Injecting noise into ~30% of examples...")
    all_examples = [_inject_noise(ex) if random.random() < 0.30 else ex
                    for ex in all_examples]

    records = [
        {
            "input_text":  build_prompt(ex["findings"]),
            "output_text": build_target(ex["root_cause"], ex["summary"]),
        }
        for ex in all_examples
    ]
    dataset = Dataset.from_list(records)
    split   = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"  Total — Train: {len(split['train'])} | Eval: {len(split['test'])}")

    # ── Step 2: load base model ───────────────────────────────────────────────
    print(f"Loading {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    # ── Step 3: apply LoRA ────────────────────────────────────────────────────
    print(f"Applying LoRA (r={args.lora_rank}, target modules: q, v)...")
    try:
        from peft import get_peft_model, LoraConfig, TaskType
        lora_config = LoraConfig(
            task_type      = TaskType.SEQ_2_SEQ_LM,
            r              = args.lora_rank,   # rank — higher = more capacity
            lora_alpha     = args.lora_rank * 2,  # scaling: alpha/r = 2 (standard)
            target_modules = ["q", "v"],       # query + value projections in T5 attention
            lora_dropout   = 0.1,
            bias           = "none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    except ImportError:
        print("WARNING: peft not installed — falling back to full fine-tuning.")
        print("Run: pip install peft")

    # ── Step 4: tokenise ──────────────────────────────────────────────────────
    print("Tokenising...")
    tokenized = split.map(
        lambda batch: tokenize(batch, tokenizer),
        batched=True,
        remove_columns=["input_text", "output_text"],
    )

    # ── Step 5: configure trainer ─────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        learning_rate               = 3e-4,
        warmup_steps                = 50,
        weight_decay                = 0.01,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        predict_with_generate       = True,
        logging_steps               = 10,
        fp16                        = False,   # fp16 caused gradient underflow on small data
        report_to                   = "none",
    )

    trainer = Seq2SeqTrainer(
        model         = model,
        args          = training_args,
        train_dataset = tokenized["train"],
        eval_dataset  = tokenized["test"],
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
    )

    # ── Step 6: train ─────────────────────────────────────────────────────────
    print("Training...")
    trainer.train()

    # ── Step 7: save ──────────────────────────────────────────────────────────
    final_dir = Path(OUTPUT_DIR) / "final"
    print(f"Saving model to {final_dir}...")
    try:
        # If LoRA was used, merge adapters back into the base model before saving
        merged = model.merge_and_unload()
        merged.save_pretrained(str(final_dir))
    except AttributeError:
        model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # ── Step 8: optionally push to HuggingFace Hub ───────────────────────────
    if args.push:
        import os
        token = args.hub_token or os.environ.get("HF_TOKEN")
        if not token:
            print("ERROR: --hub-token or HF_TOKEN env var required for push.")
        else:
            print(f"Pushing to {HF_REPO}...")
            try:
                merged.push_to_hub(HF_REPO, token=token)
            except NameError:
                model.push_to_hub(HF_REPO, token=token)
            tokenizer.push_to_hub(HF_REPO, token=token)
            print("Push complete.")

    print("Done.")


if __name__ == "__main__":
    main()