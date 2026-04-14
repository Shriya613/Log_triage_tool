"""
train.py
Fine-tunes google/flan-t5-base on surgical device log triage data.

Data sources:
  1. Synthetic generators  — 500 examples, 7 scenario types, 5-8 summary
     variants each so the model learns to reason, not memorise a template.
  2. External CSV (optional) — AI4I 2020 Predictive Maintenance dataset from
     Kaggle (search "AI4I 2020 predictive maintenance"). Pass the downloaded
     CSV with --extra-data path/to/ai4i2020.csv to add ~800 real failure rows.

Run on Colab after cloning the repo:
    python train.py                                     # synthetic only
    python train.py --extra-data ai4i2020.csv           # synthetic + external
    python train.py --push --hub-token YOUR_TOKEN       # train + publish
"""

import argparse
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

BASE_MODEL = "google/flan-t5-base"
OUTPUT_DIR  = "model_output"
HF_REPO     = "Shriya613/surgical-log-triage"

# Expanded component list for more varied findings
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

# ── Synthetic data generators ──────────────────────────────────────────────────
# Each function builds the findings dict, then picks ONE summary from a list of
# 5-8 linguistically varied options. Same facts, different phrasing — this is
# what forces the model to learn the reasoning pattern, not a single template.

def _make_nominal():
    info_count = random.randint(5, 30)
    summaries = [
        "Risk level is NOMINAL with no errors or warnings detected. "
        "All components are operating within expected parameters. "
        "No action is required.",

        f"The device log recorded {info_count} informational event(s) with no "
        "warnings, errors, or critical failures. All subsystems are functioning "
        "normally — no intervention required.",

        "Log analysis complete. Zero anomalies detected across all subsystems. "
        "Event distribution is entirely informational. "
        "Device is cleared for continued operation.",

        f"Clean run: {info_count} INFO events, zero flagged issues. "
        "No component shows signs of stress or failure. "
        "Proceed with normal operations.",

        "All logged events are at INFO level with no anomalies across any "
        "subsystem. Risk is NOMINAL. No maintenance action is needed.",

        "No issues found in this log. The device completed its cycle with only "
        "informational events. Risk assessment: NOMINAL — no follow-up required.",
    ]
    return {
        "findings": {
            "risk_level": "NOMINAL",
            "level_counts": {"INFO": info_count},
            "total_flagged_events": 0,
            "critical_events": [],
            "repeated_error_codes": {},
            "repeated_messages": {},
            "component_error_breakdown": {},
        },
        "summary": random.choice(summaries),
    }


def _make_low():
    comp       = random.choice(COMPONENTS)
    warn_count = random.randint(5, 15)
    info_count = random.randint(3, 12)
    summaries = [
        f"Risk level is LOW based on {warn_count} WARNING events in the {comp} "
        "subsystem. No errors or critical failures have occurred, but the "
        "warning pattern warrants monitoring. Schedule a diagnostic check if "
        "it persists.",

        f"{warn_count} warnings recorded in the {comp} component — below error "
        "threshold but above normal. Risk is LOW. Review the {comp} subsystem "
        "logs at the next scheduled maintenance window.",

        f"Elevated warning activity detected in {comp} ({warn_count} events). "
        "Risk level assessed as LOW. No immediate action is required, but "
        "continued operation should be monitored closely.",

        f"The {comp} subsystem generated {warn_count} WARNING events during this "
        "session. While no hard failures occurred, this volume exceeds baseline "
        "and should be reviewed before the next operational cycle.",

        f"Risk: LOW. {warn_count} warnings in {comp}, {info_count} informational "
        "events elsewhere. The device is operational but the {comp} warning "
        "count suggests early-stage degradation — flag for inspection.",

        f"Warnings in {comp} ({warn_count}) have crossed the LOW-risk threshold. "
        "No errors or critical events recorded. Recommend a non-urgent diagnostic "
        "of the {comp} component to rule out progressive fault.",
    ]
    return {
        "findings": {
            "risk_level": "LOW",
            "level_counts": {"INFO": info_count, "WARNING": warn_count},
            "total_flagged_events": 0,
            "critical_events": [],
            "repeated_error_codes": {},
            "repeated_messages": {},
            "component_error_breakdown": {},
        },
        "summary": random.choice(summaries),
    }


def _make_medium():
    comp  = random.choice(COMPONENTS)
    count = random.randint(3, 8)
    code  = ERROR_CODES[comp]
    info  = random.randint(2, 10)
    summaries = [
        f"Risk level is MEDIUM with {count} ERROR events in the {comp} subsystem "
        f"(code {code}). No critical failures occurred, but the error count "
        f"exceeds threshold. Diagnostic review of {comp} is recommended before "
        "the next operational cycle.",

        f"{count} errors flagged in the {comp} component (code {code}). Risk is "
        "MEDIUM — the device is still operable but degraded. Investigate the "
        f"{comp} subsystem before the next use.",

        f"Error code {code} appeared {count} times in {comp}, pushing risk to "
        "MEDIUM. No critical events, but recurring errors in a single component "
        "suggest a developing fault. Schedule a targeted inspection.",

        f"The {comp} subsystem reported {count} ERROR-level events (code {code}) "
        "during this session. Risk: MEDIUM. While the device completed its cycle, "
        f"the {comp} component requires review prior to next deployment.",

        f"MEDIUM risk: {count} errors from {comp} (code {code}). This is above "
        "the warning threshold but below critical. Recommend a diagnostic sweep "
        f"of the {comp} module — do not ignore if the count rises next session.",

        f"Log shows {count} {comp} errors (code {code}) alongside {info} "
        "informational events. Risk assessed as MEDIUM. No escalation needed now, "
        f"but {comp} should be checked at the earliest opportunity.",
    ]
    return {
        "findings": {
            "risk_level": "MEDIUM",
            "level_counts": {"INFO": info, "ERROR": count},
            "total_flagged_events": count,
            "critical_events": [],
            "repeated_error_codes": {},
            "repeated_messages": {},
            "component_error_breakdown": {comp: count},
        },
        "summary": random.choice(summaries),
    }


def _make_high_repeated():
    comp  = random.choice(COMPONENTS)
    code  = ERROR_CODES[comp]
    count = random.randint(3, 8)
    info  = random.randint(1, 6)
    summaries = [
        f"Risk level is HIGH: error code {code} fired {count} times in the {comp} "
        "subsystem. Repeated identical codes indicate a persistent fault, not a "
        f"transient event. Take {comp} offline and do not clear the log until "
        "root cause is confirmed.",

        f"HIGH risk — {comp} is stuck in a repeated error loop (code {code}, "
        f"{count} occurrences). This pattern points to an unresolved hardware or "
        f"firmware fault. Isolate the {comp} component immediately.",

        f"Error code {code} has repeated {count} times in {comp} — a clear "
        "indicator of a persistent failure mode rather than a one-off fault. "
        f"Risk is HIGH. The {comp} subsystem must be inspected before the device "
        "is returned to service.",

        f"The {comp} subsystem is generating the same error code ({code}) "
        f"repeatedly ({count} times). Risk: HIGH. Persistent errors in a single "
        "code space suggest a stuck state or loop condition — escalate to "
        "engineering for root cause analysis.",

        f"HIGH risk detected: {count} instances of error {code} from {comp}. "
        "Repeated firing of one error code is a strong signal of a systematic "
        f"failure in the {comp} component. Do not resume operation until the "
        "fault is diagnosed and cleared.",

        f"Log analysis flagged {count} occurrences of error code {code} from "
        f"the {comp} subsystem (risk: HIGH). A fault causing the same code to "
        "repeat is not self-resolving. Remove the device from service and "
        "inspect the {comp} module.",
    ]
    return {
        "findings": {
            "risk_level": "HIGH",
            "level_counts": {"INFO": info, "ERROR": count},
            "total_flagged_events": count,
            "critical_events": [],
            "repeated_error_codes": {code: count},
            "repeated_messages": {},
            "component_error_breakdown": {comp: count},
        },
        "summary": random.choice(summaries),
    }


def _make_high_critical():
    comp  = random.choice(COMPONENTS)
    code  = ERROR_CODES[comp]
    count = random.randint(1, 3)
    info  = random.randint(1, 5)
    summaries = [
        f"Risk level is HIGH: {count} CRITICAL event(s) in the {comp} subsystem "
        f"(code {code}). CRITICAL events indicate a hard failure — the device "
        "must not be operated until the fault is resolved. Take offline "
        f"immediately and escalate to the {comp} engineering team.",

        f"CRITICAL failure in {comp} (code {code}, {count} event(s)). Risk is "
        "HIGH. This is a hard stop — the device should be removed from service "
        "immediately and the {comp} module sent for inspection.",

        f"{count} CRITICAL-level event(s) detected in the {comp} subsystem "
        f"(error code {code}). This requires immediate action: power down the "
        f"device, quarantine the log, and notify the {comp} engineering team "
        "before any further use.",

        f"Hard failure confirmed: {comp} subsystem raised {count} CRITICAL alert(s) "
        f"(code {code}). Risk: HIGH. CRITICAL events are non-recoverable in the "
        "field — escalate now. Do not attempt a soft reset without engineering "
        "sign-off.",

        f"Risk HIGH — {comp} component experienced {count} CRITICAL failure(s) "
        f"(code {code}). Immediate offline required. CRITICAL events in this "
        "subsystem indicate the device has exceeded safe operating parameters "
        "and must be reviewed before next use.",

        f"The {comp} subsystem has entered a CRITICAL failure state ({count} "
        f"event(s), code {code}). This is the highest severity flag — cease "
        "operation, preserve the log, and engage the {comp} support team for "
        "a full failure analysis.",
    ]
    return {
        "findings": {
            "risk_level": "HIGH",
            "level_counts": {"INFO": info, "CRITICAL": count},
            "total_flagged_events": count,
            "critical_events": [
                {"level": "CRITICAL", "code": code, "component": comp,
                 "message": f"{comp.capitalize()} subsystem failure"}
                for _ in range(count)
            ],
            "repeated_error_codes": {},
            "repeated_messages": {},
            "component_error_breakdown": {comp: count},
        },
        "summary": random.choice(summaries),
    }


def _make_high_multi():
    comp_a, comp_b = random.sample(COMPONENTS, 2)
    code_a, code_b = ERROR_CODES[comp_a], ERROR_CODES[comp_b]
    err_count = random.randint(2, 5)
    summaries = [
        f"Risk level is HIGH: CRITICAL failure in {comp_a} (code {code_a}) and "
        f"repeated error code {code_b} in {comp_b} ({err_count}x). "
        "Multi-component failure suggests a shared root cause — possibly a power "
        f"rail or communication bus. Isolate both {comp_a} and {comp_b} and run "
        "a full system diagnostic.",

        f"Two subsystems affected — {comp_a} (CRITICAL, code {code_a}) and "
        f"{comp_b} (repeated error {code_b}, {err_count} times). Risk: HIGH. "
        "Simultaneous failures across components are rarely independent; "
        "investigate shared infrastructure before replacing individual parts.",

        f"HIGH risk: cascading failure across {comp_a} and {comp_b}. {comp_a} "
        f"raised a CRITICAL event (code {code_a}); {comp_b} logged {err_count} "
        f"repeated errors (code {code_b}). Take the full device offline — "
        "do not operate until both subsystems are cleared.",

        f"Multi-component fault detected. {comp_a}: CRITICAL (code {code_a}). "
        f"{comp_b}: {err_count} repeated errors (code {code_b}). Risk: HIGH. "
        "This pattern is consistent with a systemic issue rather than isolated "
        "component wear. Escalate to systems engineering immediately.",

        f"Risk HIGH — {comp_a} has a CRITICAL failure and {comp_b} is in a "
        f"repeated error state ({err_count}x code {code_b}). When two subsystems "
        "fail simultaneously, assume the failure is related until proven otherwise. "
        "Full device quarantine and diagnostic required.",
    ]
    return {
        "findings": {
            "risk_level": "HIGH",
            "level_counts": {"INFO": random.randint(1, 4),
                             "ERROR": err_count, "CRITICAL": 1},
            "total_flagged_events": err_count + 1,
            "critical_events": [
                {"level": "CRITICAL", "code": code_a, "component": comp_a,
                 "message": f"{comp_a.capitalize()} failure"}
            ],
            "repeated_error_codes": {code_b: err_count},
            "repeated_messages": {},
            "component_error_breakdown": {comp_a: 1, comp_b: err_count},
        },
        "summary": random.choice(summaries),
    }


def _make_high_escalating():
    """Scenario: log shows warnings → errors → critical in sequence.
    Represents a device that degraded progressively during a session."""
    comp       = random.choice(COMPONENTS)
    code       = ERROR_CODES[comp]
    warn_count = random.randint(3, 8)
    err_count  = random.randint(2, 5)
    summaries = [
        f"Risk level is HIGH: the {comp} subsystem shows a clear escalation "
        f"pattern — {warn_count} warnings followed by {err_count} errors and a "
        "CRITICAL event. This trajectory indicates progressive failure during "
        "the session. The device must be taken offline and the full event "
        "sequence reviewed.",

        f"HIGH risk — escalating failure sequence detected in {comp} (code "
        f"{code}): {warn_count} warnings → {err_count} errors → CRITICAL. "
        "Progressive degradation of this kind is a strong predictor of "
        "imminent hardware failure. Do not restart the device without a "
        "full inspection.",

        f"The {comp} component degraded progressively this session: "
        f"{warn_count} warnings, then {err_count} errors, culminating in a "
        f"CRITICAL event (code {code}). Risk: HIGH. This is not a one-time "
        "fault — the component is failing. Replace or service before next use.",

        f"Log shows an escalating fault in {comp}: warnings ({warn_count}) → "
        f"errors ({err_count}) → CRITICAL. Risk: HIGH. When a component moves "
        "through all severity levels in a single session it indicates a "
        "progressive hardware fault. Immediate offline and inspection required.",

        f"Escalation pattern in {comp} subsystem: "
        f"{warn_count}x WARNING → {err_count}x ERROR → 1x CRITICAL (code {code}). "
        "Risk is HIGH. Treat this as a degrading component, not an isolated "
        "event — schedule replacement before the next operational cycle.",
    ]
    return {
        "findings": {
            "risk_level": "HIGH",
            "level_counts": {
                "INFO":     random.randint(2, 6),
                "WARNING":  warn_count,
                "ERROR":    err_count,
                "CRITICAL": 1,
            },
            "total_flagged_events": err_count + 1,
            "critical_events": [
                {"level": "CRITICAL", "code": code, "component": comp,
                 "message": f"{comp.capitalize()} progressive failure"}
            ],
            "repeated_error_codes": {},
            "repeated_messages": {},
            "component_error_breakdown": {comp: err_count + 1},
        },
        "summary": random.choice(summaries),
    }


# (generator_function, number_of_examples)  — total: 500
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


# ── External dataset loader (AI4I 2020 Predictive Maintenance) ─────────────────
# Download from Kaggle: search "AI4I 2020 predictive maintenance dataset"
# Direct link: kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020
#
# The CSV has columns: Product ID, Type, Air temperature [K],
# Process temperature [K], Rotational speed [rpm], Torque [Nm],
# Tool wear [min], Machine failure, TWF, HDF, PWF, OSF, RNF
#
# We map each row's failure flags to our findings format and apply
# the same varied summary generators so language stays diverse.

_AI4I_COMPONENT_MAP = {
    "TWF": ("motor",       "5001"),   # Tool Wear Failure
    "HDF": ("sensor",      "2001"),   # Heat Dissipation Failure
    "PWF": ("power",       "9001"),   # Power Failure
    "OSF": ("calibration", "3001"),   # Overstrain Failure
    "RNF": ("core",        "1001"),   # Random Failure
}


def load_ai4i_data(csv_path: str) -> list[dict]:
    """
    Convert AI4I 2020 Predictive Maintenance rows to (findings, summary) pairs.
    Rows with no machine failure become NOMINAL examples.
    Rows with one or more failure flags become HIGH examples.
    """
    try:
        import pandas as pd
    except ImportError:
        print("pandas required for external data loading — skipping.")
        return []

    path = Path(csv_path)
    if not path.exists():
        print(f"\nERROR: File not found: {csv_path}")
        print("To find where your file was uploaded in Colab, run:")
        print("  !find / -name '*.csv' 2>/dev/null | grep -i ai4i")
        print("Then re-run with the correct path, e.g.:")
        print("  !python train.py --extra-data /content/ai4i2020.csv\n")
        return []

    df = pd.read_csv(path)
    failure_cols = [c for c in ["TWF", "HDF", "PWF", "OSF", "RNF"] if c in df.columns]
    if not failure_cols:
        print(f"WARNING: expected failure columns not found in {csv_path} — skipping.")
        return []

    examples = []
    for _, row in df.iterrows():
        active = [c for c in failure_cols if int(row.get(c, 0)) == 1]

        if not active:
            # No failure → NOMINAL
            ex = _make_nominal()
        elif len(active) == 1:
            # Single failure → map to component and build HIGH critical example
            comp, code = _AI4I_COMPONENT_MAP[active[0]]
            ex = _make_high_critical()
            # Patch findings to reflect the actual component from the row
            ex["findings"]["critical_events"] = [
                {"level": "CRITICAL", "code": code, "component": comp,
                 "message": f"{comp.capitalize()} failure ({active[0]})"}
            ]
            ex["findings"]["component_error_breakdown"] = {comp: 1}
        else:
            # Multiple failures → multi-component HIGH
            ex = _make_high_multi()

        examples.append(ex)

    print(f"  Loaded {len(examples)} examples from {csv_path}")
    return examples


# ── Prompt builder (must match app/summarizer.py exactly) ─────────────────────

def build_prompt(findings: dict) -> str:
    compact = {k: v for k, v in findings.items() if k != "flagged_events"}
    return (
        "Summarize the following medical device log findings in plain English. "
        "State what failed, which components are affected, the risk level, "
        "and whether immediate action is required.\n\n"
        f"Findings:\n{json.dumps(compact, indent=2)}"
    )


# ── Tokenisation ──────────────────────────────────────────────────────────────

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
    # Pad positions in labels are set to -100 so the loss function ignores them.
    labels["input_ids"] = [
        [(t if t != tokenizer.pad_token_id else -100) for t in label]
        for label in labels["input_ids"]
    ]
    inputs["labels"] = labels["input_ids"]
    return inputs


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--push",       action="store_true",
                        help="Push the fine-tuned model to HuggingFace Hub")
    parser.add_argument("--hub-token",  type=str, default=None,
                        help="HuggingFace write token (or set HF_TOKEN env var)")
    parser.add_argument("--epochs",     type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--extra-data", type=str, default=None,
                        metavar="CSV_PATH",
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

    records = [
        {"input_text": build_prompt(ex["findings"]), "output_text": ex["summary"]}
        for ex in all_examples
    ]
    dataset = Dataset.from_list(records)
    split   = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"  Total — Train: {len(split['train'])} | Eval: {len(split['test'])}")

    # ── Step 2: load base model ───────────────────────────────────────────────
    print(f"Loading {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    # ── Step 3: tokenise ──────────────────────────────────────────────────────
    print("Tokenising...")
    tokenized = split.map(
        lambda batch: tokenize(batch, tokenizer),
        batched=True,
        remove_columns=["input_text", "output_text"],
    )

    # ── Step 4: configure trainer ─────────────────────────────────────────────
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
        fp16                        = False,
        report_to                   = "none",
    )

    trainer = Seq2SeqTrainer(
        model         = model,
        args          = training_args,
        train_dataset = tokenized["train"],
        eval_dataset  = tokenized["test"],
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
    )

    # ── Step 5: train ─────────────────────────────────────────────────────────
    print("Training...")
    trainer.train()

    # ── Step 6: save ──────────────────────────────────────────────────────────
    save_path = f"{OUTPUT_DIR}/final"
    print(f"Saving to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # ── Step 7: push to Hub (optional) ────────────────────────────────────────
    if args.push:
        import os
        token = args.hub_token or os.getenv("HF_TOKEN")
        if not token:
            print("ERROR: --push requires --hub-token TOKEN or HF_TOKEN env var.")
            return
        print(f"Pushing to {HF_REPO}...")
        model.push_to_hub(HF_REPO, token=token)
        tokenizer.push_to_hub(HF_REPO, token=token)
        print(f"Live at https://huggingface.co/{HF_REPO}")


if __name__ == "__main__":
    main()