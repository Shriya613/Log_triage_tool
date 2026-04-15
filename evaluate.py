"""
evaluate.py
Evaluate the triage pipeline on a held-out test set.

Metrics tracked:
  schema_valid accuracy  — did we correctly identify valid vs invalid schemas?
  severity accuracy      — did we assign the right severity (low/medium/high)?
  anomaly precision      — of the anomaly types we detected, how many were correct?
  anomaly recall         — of the anomaly types that should be detected, how many did we catch?
  anomaly F1             — harmonic mean of precision and recall

Test file format (JSON array):
  [
    {
      "input": { <raw device log JSON> },
      "expected": {
        "schema_valid": true,
        "severity": "high",
        "anomaly_types": ["critical_event", "repeated_error"]
      }
    },
    ...
  ]

Usage:
  python evaluate.py --test-file results/eval_set.json
  python evaluate.py --test-file results/eval_set.json --generate-sample   # create a sample set
"""

import argparse
import json
import random
from pathlib import Path

from app.parser import parse_log
from app.analyzer import analyze_log


# ── Core evaluation ────────────────────────────────────────────────────────────

def evaluate(test_cases: list[dict]) -> dict:
    """
    Run the deterministic pipeline (parser + analyzer) on each test case.
    Returns a metrics summary dict.
    """
    total = len(test_cases)
    if total == 0:
        return {"error": "No test cases provided"}

    schema_correct   = 0
    severity_correct = 0
    tp = fp = fn = 0   # anomaly type true/false positives/negatives

    per_case = []

    for i, case in enumerate(test_cases):
        parsed   = parse_log(case["input"])
        findings = analyze_log(parsed)

        expected = case["expected"]

        # Schema valid
        schema_match = parsed["schema_valid"] == expected.get("schema_valid", True)
        if schema_match:
            schema_correct += 1

        # Severity
        severity_match = findings["severity"] == expected.get("severity", "low")
        if severity_match:
            severity_correct += 1

        # Anomaly type precision/recall
        detected_types = {a["type"] for a in findings["anomalies"]}
        expected_types = set(expected.get("anomaly_types", []))
        case_tp = len(detected_types & expected_types)
        case_fp = len(detected_types - expected_types)
        case_fn = len(expected_types - detected_types)
        tp += case_tp
        fp += case_fp
        fn += case_fn

        per_case.append({
            "case":            i,
            "schema_match":    schema_match,
            "severity_match":  severity_match,
            "expected_severity": expected.get("severity"),
            "got_severity":    findings["severity"],
            "detected_types":  sorted(detected_types),
            "expected_types":  sorted(expected_types),
            "false_positives": sorted(detected_types - expected_types),
            "false_negatives": sorted(expected_types - detected_types),
        })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "total_cases":              total,
        "schema_accuracy":          round(schema_correct / total, 3),
        "severity_accuracy":        round(severity_correct / total, 3),
        "anomaly_precision":        round(precision, 3),
        "anomaly_recall":           round(recall, 3),
        "anomaly_f1":               round(f1, 3),
        "per_case":                 per_case,
    }


def print_report(metrics: dict):
    total = metrics["total_cases"]
    print(f"\n{'='*52}")
    print(f"  Triage Pipeline Evaluation ({total} test cases)")
    print(f"{'='*52}")
    print(f"  Schema valid accuracy : {metrics['schema_accuracy']:.1%}")
    print(f"  Severity accuracy     : {metrics['severity_accuracy']:.1%}")
    print(f"  Anomaly precision     : {metrics['anomaly_precision']:.3f}")
    print(f"  Anomaly recall        : {metrics['anomaly_recall']:.3f}")
    print(f"  Anomaly F1            : {metrics['anomaly_f1']:.3f}")
    print(f"{'='*52}")

    # Show failures
    failures = [c for c in metrics["per_case"]
                if not c["severity_match"] or c["false_positives"] or c["false_negatives"]]
    if failures:
        print(f"\n  Failures / mismatches ({len(failures)}):")
        for c in failures[:10]:   # cap at 10 for readability
            if not c["severity_match"]:
                print(f"    Case {c['case']:3d} | severity: "
                      f"expected={c['expected_severity']}  got={c['got_severity']}")
            if c["false_positives"]:
                print(f"    Case {c['case']:3d} | FP anomaly types: {c['false_positives']}")
            if c["false_negatives"]:
                print(f"    Case {c['case']:3d} | FN anomaly types: {c['false_negatives']}")
        if len(failures) > 10:
            print(f"    ... and {len(failures) - 10} more")
    else:
        print("\n  No failures — all cases matched expected output.")
    print()


# ── Sample test set generator ──────────────────────────────────────────────────

def generate_sample_eval_set(n: int = 50) -> list[dict]:
    """
    Build a small labelled eval set using known-good inputs.
    Useful when you don't have a hand-labelled set yet.
    """
    import random
    random.seed(99)

    cases = []

    # --- Clean valid log → schema_valid=True, severity=low
    for _ in range(n // 5):
        cases.append({
            "input": {
                "device_id": "DEV-EVAL-CLEAN",
                "timestamp": "2024-01-01T00:00:00Z",
                "events": [
                    {"event_id": "E001", "timestamp": "2024-01-01T00:00:01Z",
                     "level": "INFO", "message": "Startup complete",
                     "code": "1000", "component": "core"},
                ]
            },
            "expected": {"schema_valid": True, "severity": "low", "anomaly_types": []},
        })

    # --- High-risk critical log → schema_valid=True, severity=high, critical_event
    for _ in range(n // 5):
        cases.append({
            "input": {
                "device_id": "DEV-EVAL-CRIT",
                "timestamp": "2024-01-01T01:00:00Z",
                "events": [
                    {"event_id": f"E{i:03}", "timestamp": "2024-01-01T01:00:01Z",
                     "level": "CRITICAL", "message": "Motor failure",
                     "code": "5001", "component": "motor"}
                    for i in range(2)
                ]
            },
            "expected": {"schema_valid": True, "severity": "high",
                         "anomaly_types": ["critical_event"]},
        })

    # --- Repeated errors → severity=high, repeated_error
    for _ in range(n // 5):
        cases.append({
            "input": {
                "device_id": "DEV-EVAL-REPEAT",
                "timestamp": "2024-01-01T02:00:00Z",
                "events": [
                    {"event_id": f"E{i:03}", "timestamp": f"2024-01-01T02:00:0{i+1}Z",
                     "level": "ERROR", "message": "Calibration failed",
                     "code": "3001", "component": "calibration"}
                    for i in range(4)
                ]
            },
            "expected": {"schema_valid": True, "severity": "high",
                         "anomaly_types": ["repeated_error"]},
        })

    # --- Missing required field → schema_valid=False
    for _ in range(n // 5):
        cases.append({
            "input": {
                "timestamp": "2024-01-01T03:00:00Z",
                "events": [
                    {"event_id": "E001", "timestamp": "2024-01-01T03:00:01Z",
                     "level": "INFO", "message": "Test"}
                ]
                # missing device_id
            },
            "expected": {"schema_valid": False, "severity": "low",
                         "anomaly_types": ["missing_field"]},
        })

    # --- Warning cluster → schema_valid=True, severity=low, warning_cluster
    for _ in range(n // 5):
        cases.append({
            "input": {
                "device_id": "DEV-EVAL-WARN",
                "timestamp": "2024-01-01T04:00:00Z",
                "events": [
                    {"event_id": f"E{i:03}", "timestamp": f"2024-01-01T04:00:{i+1:02d}Z",
                     "level": "WARNING", "message": "Sensor drift",
                     "code": "2001", "component": "sensor"}
                    for i in range(6)
                ]
            },
            "expected": {"schema_valid": True, "severity": "low",
                         "anomaly_types": ["warning_cluster"]},
        })

    return cases


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate the triage pipeline")
    parser.add_argument("--test-file", type=str, default=None,
                        help="Path to JSON test cases file")
    parser.add_argument("--generate-sample", action="store_true",
                        help="Generate a sample eval set and save to results/eval_set.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Save full metrics JSON to this file")
    args = parser.parse_args()

    if args.generate_sample:
        out_path = Path("results/eval_set.json")
        out_path.parent.mkdir(exist_ok=True)
        cases = generate_sample_eval_set()
        out_path.write_text(json.dumps(cases, indent=2))
        print(f"Sample eval set written to {out_path} ({len(cases)} cases)")
        # Also evaluate it immediately
        metrics = evaluate(cases)
        print_report(metrics)
        return

    if not args.test_file:
        parser.error("Provide --test-file or --generate-sample")

    path = Path(args.test_file)
    if not path.exists():
        print(f"ERROR: test file not found: {path}")
        return

    test_cases = json.loads(path.read_text())
    metrics    = evaluate(test_cases)
    print_report(metrics)

    if args.output:
        Path(args.output).write_text(json.dumps(metrics, indent=2))
        print(f"Full metrics saved to {args.output}")


if __name__ == "__main__":
    main()