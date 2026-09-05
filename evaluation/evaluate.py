import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, Optional

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.services.verifier import VerificationPipeline
from backend.schemas import VerdictType

BENCHMARK_FILE = Path(__file__).parent / "benchmark_data.json"
RESULTS_CACHE_FILE = Path(__file__).parent / "benchmark_results.json"


def run_benchmark(
    benchmark_path: Optional[Path] = None,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Executes the VERDICT benchmark suite against ground-truth test cases.
    Computes accuracy, precision, recall, F1 per class, macro F1, confusion matrix,
    evidence retrieval precision, and execution timing metrics.
    """
    bench_file = benchmark_path or BENCHMARK_FILE
    if not bench_file.exists():
        raise FileNotFoundError(f"Benchmark dataset not found at {bench_file}")

    with open(bench_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_cases = data.get("test_cases", [])
    source_content = data.get("source_content", "").encode("utf-8")
    source_filename = data.get("source_document", "source.txt")
    dataset_name = data.get("dataset_name", "VERDICT Benchmark Suite")

    pipeline = VerificationPipeline()

    start_suite_time = time.perf_counter()
    correct_verdicts = 0
    retrieval_hits = 0
    retrieval_eligible = 0

    classes = [VerdictType.SUPPORTED.value, VerdictType.REFUTED.value, VerdictType.UNVERIFIED.value]

    # Confusion matrix structure: true_label -> pred_label -> count
    confusion_matrix = {
        true_cls: {pred_cls: 0 for pred_cls in classes}
        for true_cls in classes
    }

    class_counts = {
        c: {"tp": 0, "fp": 0, "fn": 0, "total_gt": 0}
        for c in classes
    }

    detailed_cases = []

    for tc in test_cases:
        cid = tc["claim_id"]
        claim_text = tc["claim"]
        gt_verdict = tc["ground_truth_verdict"]
        expected_kw = tc.get("expected_evidence_keywords", [])

        t0 = time.perf_counter()
        cert = pipeline.run_verification(
            draft_text=claim_text,
            source_files=[(source_filename, source_content)],
            top_k=3
        )
        case_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        pred_claim = cert.claims[0] if cert.claims else None
        pred_verdict = pred_claim.verdict.value if pred_claim else VerdictType.UNVERIFIED.value

        # Check retrieval precision
        hit = False
        retrieved_text = ""
        if expected_kw:
            retrieval_eligible += 1
            retrieved_text = " ".join([e.text for e in pred_claim.evidence]) if pred_claim else ""
            hit = any(kw.lower() in retrieved_text.lower() for kw in expected_kw)
            if hit:
                retrieval_hits += 1

        # Accuracy & Confusion Matrix
        is_correct = (pred_verdict == gt_verdict)
        if is_correct:
            correct_verdicts += 1
            class_counts[gt_verdict]["tp"] += 1
        else:
            class_counts[gt_verdict]["fn"] += 1
            if pred_verdict in class_counts:
                class_counts[pred_verdict]["fp"] += 1

        if gt_verdict in confusion_matrix and pred_verdict in confusion_matrix[gt_verdict]:
            confusion_matrix[gt_verdict][pred_verdict] += 1

        class_counts[gt_verdict]["total_gt"] += 1

        detailed_cases.append({
            "claim_id": cid,
            "claim": claim_text,
            "ground_truth": gt_verdict,
            "predicted": pred_verdict,
            "is_correct": is_correct,
            "retrieval_hit": hit if expected_kw else None,
            "evidence_text": pred_claim.evidence[0].text if (pred_claim and pred_claim.evidence) else "",
            "processing_time_ms": case_time_ms
        })

    total_time_seconds = round(time.perf_counter() - start_suite_time, 3)
    total = len(test_cases)
    accuracy = round((correct_verdicts / total) * 100, 2) if total > 0 else 0.0
    retrieval_prec = round((retrieval_hits / retrieval_eligible) * 100, 2) if retrieval_eligible > 0 else 0.0
    avg_time_ms = round((total_time_seconds * 1000.0) / total, 2) if total > 0 else 0.0

    # Per-class precision, recall, f1
    class_metrics = {}
    f1_list = []
    for c in classes:
        tp = class_counts[c]["tp"]
        fp = class_counts[c]["fp"]
        fn = class_counts[c]["fn"]
        support = class_counts[c]["total_gt"]

        prec = round((tp / (tp + fp)) * 100, 2) if (tp + fp) > 0 else 0.0
        rec = round((tp / (tp + fn)) * 100, 2) if (tp + fn) > 0 else 0.0
        f1 = round((2 * (prec * rec) / (prec + rec)), 2) if (prec + rec) > 0 else 0.0

        f1_list.append(f1)
        class_metrics[c] = {
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "support": support,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        }

    macro_f1 = round(sum(f1_list) / len(f1_list), 2) if f1_list else 0.0

    result = {
        "dataset_name": dataset_name,
        "total_test_cases": total,
        "correct_verdicts": correct_verdicts,
        "accuracy": accuracy,
        "retrieval_precision": retrieval_prec,
        "macro_f1": macro_f1,
        "total_time_seconds": total_time_seconds,
        "avg_time_per_claim_ms": avg_time_ms,
        "class_metrics": class_metrics,
        "confusion_matrix": confusion_matrix,
        "test_cases": detailed_cases,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    if save_results:
        try:
            with open(RESULTS_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to cache benchmark results: {e}")

    return result


def get_or_run_benchmark(fresh: bool = False) -> Dict[str, Any]:
    """
    Returns cached benchmark results if available and fresh=False;
    otherwise runs the benchmark and caches the new results.
    """
    if not fresh and RESULTS_CACHE_FILE.exists():
        try:
            with open(RESULTS_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return run_benchmark(save_results=True)


def print_cli_summary(res: Dict[str, Any]):
    print("=" * 72)
    print(f"📊 VERDICT Evaluation Suite: {res.get('dataset_name')}")
    print(f"Total Test Cases: {res.get('total_test_cases')}")
    print("=" * 72)

    for tc in res.get("test_cases", []):
        sym = "✓" if tc["is_correct"] else "✗"
        print(f"[{sym}] {tc['claim_id']}: GroundTruth={tc['ground_truth']:<10} | Pred={tc['predicted']:<10} | Claim: \"{tc['claim'][:40]}...\" ({tc['processing_time_ms']}ms)")

    print("\n" + "=" * 72)
    print("🏆 EVALUATION SUMMARY")
    print("=" * 72)
    print(f"Overall Accuracy:                    {res['accuracy']:.1f}% ({res['correct_verdicts']}/{res['total_test_cases']})")
    print(f"Evidence Retrieval Precision (Top-K):{res['retrieval_precision']:.1f}%")
    print(f"Macro F1-Score:                      {res['macro_f1']:.1f}%")
    print(f"Total Benchmark Execution Time:      {res['total_time_seconds']:.2f}s")
    print(f"Average Time per Claim:              {res['avg_time_per_claim_ms']:.1f}ms")
    print("-" * 72)
    print(f"{'Class':<14} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 72)

    for c, m in res.get("class_metrics", {}).items():
        print(f"{c:<14} | {m['precision']:>8.1f}% | {m['recall']:>8.1f}% | {m['f1_score']:>8.1f}% | {m['support']:>8}")

    print("=" * 72)
    print("Confusion Matrix (Ground Truth Rows x Predicted Columns):")
    print(f"{'Actual':<14} | {'Pred SUPPORTED':<15} | {'Pred REFUTED':<13} | {'Pred UNVERIFIED':<15}")
    print("-" * 72)
    cm = res.get("confusion_matrix", {})
    for true_c in ["SUPPORTED", "REFUTED", "UNVERIFIED"]:
        s_c = cm.get(true_c, {}).get("SUPPORTED", 0)
        r_c = cm.get(true_c, {}).get("REFUTED", 0)
        u_c = cm.get(true_c, {}).get("UNVERIFIED", 0)
        print(f"{true_c:<14} | {s_c:>15} | {r_c:>13} | {u_c:>15}")
    print("=" * 72)


if __name__ == "__main__":
    results = run_benchmark()
    print_cli_summary(results)
