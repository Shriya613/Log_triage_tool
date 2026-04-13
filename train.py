"""
train.py
Fine-tunes google/flan-t5-base on synthetic surgical device log triage data.

Run on Colab after cloning the repo:
    python train.py                              # train and save locally
    python train.py --push --hub-token YOUR_TOKEN  # train and push to HuggingFace Hub
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

COMPONENTS = ["motor", "calibration", "sensor", "power", "navigation", "safety", "core"]
ERROR_CODES = {
    "motor":       "5001",
    "calibration": "3001",
    "sensor":      "2001",
    "power":       "9001",
    "navigation":  "4001",
    "safety":      "5002",
    "core":        "1001",
}

# ── Synthetic data generators ──────────────────────────────────────────────────
# Each function returns one (findings dict, summary string) training example.
# Findings mirror exactly what analyzer.py produces so the model sees realistic input.

def _make_nominal():
    return {
        "findings": {
            "risk_level": "NOMINAL",
            "level_counts": {"INFO": random.randint(5, 20)},
            "total_flagged_events": 0,
            "critical_events": [],
            "repeated_error_codes": {},
            "repeated_messages": {},
            "component_error_breakdown": {},
        },
        "summary": (
            "Risk level is NOMINAL with no errors or warnings detected. "
            "All events are informational and no components have flagged anomalies. "
            "No action is required at this time."
        )
    }


def _make_low():
    comp = random.choice(COMPONENTS)
    warn_count = random.randint(5, 10)
    return {
        "findings": {
            "risk_level": "LOW",
            "level_counts": {"INFO": random.randint(3, 10), "WARNING": warn_count},
            "total_flagged_events": 0,
            "critical_events": [],
            "repeated_error_codes": {},
            "repeated_messages": {},
            "component_error_breakdown": {},
        },
        "summary": (
            f"Risk level is LOW based on {warn_count} WARNING events, "
            f"primarily in the {comp} subsystem. "
            "No critical or error-level failures have occurred, "
            "but the warning pattern warrants monitoring. "
            "Schedule a diagnostic check if the pattern persists."
        )
    }


def _make_medium():
    comp  = random.choice(COMPONENTS)
    count = random.randint(3, 6)
    code  = ERROR_CODES[comp]
    return {
        "findings": {
            "risk_level": "MEDIUM",
            "level_counts": {"INFO": random.randint(2, 8), "ERROR": count},
            "total_flagged_events": count,
            "critical_events": [],
            "repeated_error_codes": {},
            "repeated_messages": {},
            "component_error_breakdown": {comp: count},
        },
        "summary": (
            f"Risk level is MEDIUM with {count} ERROR events in the {comp} subsystem (code {code}). "
            "No critical failures have occurred, but the error count exceeds the warning threshold. "
            f"Diagnostic review of the {comp} component is recommended before the next operational cycle."
        )
    }


def _make_high_repeated():
    comp  = random.choice(COMPONENTS)
    code  = ERROR_CODES[comp]
    count = random.randint(3, 6)
    return {
        "findings": {
            "risk_level": "HIGH",
            "level_counts": {"INFO": random.randint(1, 5), "ERROR": count},
            "total_flagged_events": count,
            "critical_events": [],
            "repeated_error_codes": {code: count},
            "repeated_messages": {},
            "component_error_breakdown": {comp: count},
        },
        "summary": (
            f"Risk level is HIGH due to repeated error code {code} appearing {count} times "
            f"in the {comp} subsystem. "
            "Repeated identical error codes indicate a persistent failure, not a transient fault. "
            f"Take the {comp} component offline for inspection — "
            "do not clear the log until root cause is confirmed."
        )
    }


def _make_high_critical():
    comp  = random.choice(COMPONENTS)
    code  = ERROR_CODES[comp]
    count = random.randint(1, 3)
    return {
        "findings": {
            "risk_level": "HIGH",
            "level_counts": {"INFO": random.randint(1, 4), "CRITICAL": count},
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
        "summary": (
            f"Risk level is HIGH with {count} CRITICAL event(s) in the {comp} subsystem (code {code}). "
            "CRITICAL events indicate a hard failure requiring immediate attention "
            "before the device can be safely operated. "
            f"Take the device offline immediately and escalate to the {comp} engineering team."
        )
    }


def _make_high_multi():
    comp_a, comp_b = random.sample(COMPONENTS, 2)
    code_a, code_b = ERROR_CODES[comp_a], ERROR_CODES[comp_b]
    return {
        "findings": {
            "risk_level": "HIGH",
            "level_counts": {"INFO": random.randint(1, 4), "ERROR": 3, "CRITICAL": 1},
            "total_flagged_events": 4,
            "critical_events": [
                {"level": "CRITICAL", "code": code_a, "component": comp_a,
                 "message": f"{comp_a.capitalize()} failure"}
            ],
            "repeated_error_codes": {code_b: 3},
            "repeated_messages": {},
            "component_error_breakdown": {comp_a: 1, comp_b: 3},
        },
        "summary": (
            f"Risk level is HIGH with a CRITICAL failure in the {comp_a} subsystem "
            f"and repeated error code {code_b} in the {comp_b} subsystem. "
            "Multi-component failures suggest a systemic issue, "
            "possibly related to a shared power rail or communication bus. "
            f"Isolate both {comp_a} and {comp_b} and perform a full system diagnostic immediately."
        )
    }


# (generator_function, number_of_examples_to_produce)
GENERATORS = [
    (_make_nominal,       30),
    (_make_low,           30),
    (_make_medium,        40),
    (_make_high_repeated, 50),
    (_make_high_critical, 50),
    (_make_high_multi,    30),
]


def generate_dataset(seed: int = 42) -> list[dict]:
    """Build and shuffle all synthetic examples."""
    random.seed(seed)
    examples = []
    for fn, count in GENERATORS:
        for _ in range(count):
            examples.append(fn())
    random.shuffle(examples)
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
    # Replace pad token IDs in labels with -100.
    # The cross-entropy loss ignores -100 positions so padding doesn't
    # contribute to the loss and distort training.
    labels["input_ids"] = [
        [(t if t != tokenizer.pad_token_id else -100) for t in label]
        for label in labels["input_ids"]
    ]
    inputs["labels"] = labels["input_ids"]
    return inputs


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--push",      action="store_true",
                        help="Push the fine-tuned model to HuggingFace Hub")
    parser.add_argument("--hub-token", type=str, default=None,
                        help="HuggingFace write token (or set HF_TOKEN env var)")
    parser.add_argument("--epochs",    type=int, default=5)
    parser.add_argument("--batch-size",type=int, default=8)
    args = parser.parse_args()

    # ── Step 1: generate data ─────────────────────────────────────────────────
    print("Generating synthetic training data...")
    raw     = generate_dataset()
    records = [
        {"input_text": build_prompt(ex["findings"]), "output_text": ex["summary"]}
        for ex in raw
    ]
    dataset = Dataset.from_list(records)
    split   = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"  Train: {len(split['train'])} | Eval: {len(split['test'])}")

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
        output_dir              = OUTPUT_DIR,
        num_train_epochs        = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        learning_rate           = 3e-4,
        warmup_steps            = 50,
        weight_decay            = 0.01,
        eval_strategy           = "epoch",
        save_strategy           = "epoch",
        load_best_model_at_end  = True,
        predict_with_generate   = True,
        logging_steps           = 10,
        fp16                    = False,   # flip to True on Colab T4/A100
        report_to               = "none",
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