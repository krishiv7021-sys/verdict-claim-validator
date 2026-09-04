import os
import sys
import json
from pathlib import Path
from collections import defaultdict

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.verifier import VerificationPipeline
from backend.schemas import VerdictType


def run_benchmark():
    benchmark_file = Path(__file__).parent / "benchmark_data.json"
    if not benchmark_file.exists():
        print(f"Error: {benchmark_file} not found.")
        return

    with open(benchmark_file, "r") as f:
        data = json.load(f)

    test_cases = data.get("test_cases", [])
    source_content = data.get("source_content", "").encode("utf-8")
    source_filename = data.get("source_document", "source.txt")

    print("=" * 70)
    print(f"📊 Running VERDICT Evaluation Suite: {data.get('dataset_name', '')}")
    print(f"Total Test Cases: {len(test_cases)}")
    print("=" * 70)

    pipeline = VerificationPipeline()

    correct_verdicts = 0
    retrieval_hits = 0
    retrieval_eligible = 0

    class_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "total_gt": 0})

    for tc in test_cases:
        cid = tc["claim_id"]
        claim_text = tc["claim"]
        gt_verdict = tc["ground_truth_verdict"]
        expected_kw = tc.get("expected_evidence_keywords", [])

        # Run verification through pipeline
        cert = pipeline.run_verification(
            draft_text=claim_text,
            source_files=[(source_filename, source_content)],
            top_k=3
        )

        pred_claim = cert.claims[0] if cert.claims else None
        pred_verdict = pred_claim.verdict.value if pred_claim else "UNVERIFIED"

        # Check retrieval precision
        if expected_kw:
            retrieval_eligible += 1
            retrieved_text = " ".join([e.text for e in pred_claim.evidence]) if pred_claim else ""
            hit = any(kw.lower() in retrieved_text.lower() for kw in expected_kw)
            if hit:
                retrieval_hits += 1

        # Accuracy
        is_correct = (pred_verdict == gt_verdict)
        if is_correct:
            correct_verdicts += 1
            class_counts[gt_verdict]["tp"] += 1
        else:
            class_counts[gt_verdict]["fn"] += 1
            class_counts[pred_verdict]["fp"] += 1

        class_counts[gt_verdict]["total_gt"] += 1

        status_symbol = "✓" if is_correct else "✗"
        print(f"[{status_symbol}] {cid}: GroundTruth={gt_verdict:<10} | Pred={pred_verdict:<10} | Claim: \"{claim_text[:40]}...\"")

    total = len(test_cases)
    accuracy = (correct_verdicts / total) * 100 if total > 0 else 0
    retrieval_prec = (retrieval_hits / retrieval_eligible) * 100 if retrieval_eligible > 0 else 0

    print("\n" + "=" * 70)
    print("🏆 EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Overall Claim Verification Accuracy: {accuracy:.1f}% ({correct_verdicts}/{total})")
    print(f"Evidence Retrieval Precision (Top-K): {retrieval_prec:.1f}% ({retrieval_hits}/{retrieval_eligible})")
    print("-" * 70)
    print(f"{'Class':<14} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 70)

    for c in ["SUPPORTED", "REFUTED", "UNVERIFIED"]:
        tp = class_counts[c]["tp"]
        fp = class_counts[c]["fp"]
        fn = class_counts[c]["fn"]
        support = class_counts[c]["total_gt"]

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        print(f"{c:<14} | {prec*100:>8.1f}% | {rec*100:>8.1f}% | {f1*100:>8.1f}% | {support:>8}")

    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
