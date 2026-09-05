#!/usr/bin/env python3
"""
VERDICT: Advanced Verification Demonstration Script
Demonstrates:
1. Authority-Weighted Retrieval (Statutory vs Policy vs Internal vs Reference)
2. Cross-Source Conflict Detection (Contradictions across source documents)
3. Cryptographic Audit Certificate with Timing Metrics and SHA-256
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.services.verifier import VerificationPipeline
from backend.schemas import SourceAuthorityLevel


def main():
    print("=" * 75)
    print("⚖️  VERDICT — Advanced Verification Demonstration")
    print("=" * 75)

    pipeline = VerificationPipeline()

    # Scenario:
    # Source A (Statutory Act, Authority: STATUTORY / OFFICIAL, wt: 1.00):
    #   "The statutory termination notice period is strictly 7 days."
    # Source B (Employee Handbook, Authority: INTERNAL / ORGANIZATIONAL, wt: 0.94):
    #   "The mandatory contract termination notice period is 10 business days."
    # Source C (Consultant Blog, Authority: REFERENCE / WEB, wt: 0.90):
    #   "Standard notice periods in industry generally range around 30 days."

    source_a = b"""SECTION 14 - TERMINATION NOTICE
The statutory termination notice period is strictly 7 days for all standard contracts."""
    
    source_b = b"""HANDBOOK SECTION 4.2 - NOTICE PERIODS
The mandatory contract termination notice period is 10 business days."""

    source_c = b"""INDUSTRY PRACTICE GUIDE
Standard notice periods in industry generally range around 30 days."""

    draft = """The statutory termination notice period is 7 days. Standard industry notice is 30 days."""

    print("\n1. Ingesting Multi-Tier Sources with Authority Levels:")
    print("   - [Statutory_Act.txt]    Authority: STATUTORY / OFFICIAL     (Weight: 1.00)")
    print("   - [Employee_Handbook.txt]Authority: INTERNAL / ORGANIZATIONAL(Weight: 0.94)")
    print("   - [Industry_Guide.txt]   Authority: REFERENCE / WEB          (Weight: 0.90)")

    authorities = {
        "Statutory_Act.txt": SourceAuthorityLevel.STATUTORY.value,
        "Employee_Handbook.txt": SourceAuthorityLevel.INTERNAL.value,
        "Industry_Guide.txt": SourceAuthorityLevel.REFERENCE.value,
    }

    print("\n2. Executing Verification Pipeline with Authority Weighting & Conflict Detection...")
    cert = pipeline.run_verification(
        draft_text=draft,
        source_files=[
            ("Statutory_Act.txt", source_a),
            ("Employee_Handbook.txt", source_b),
            ("Industry_Guide.txt", source_c)
        ],
        source_authorities=authorities,
        top_k=3
    )

    print("\n3. Verification Summary:")
    print(f"   - Certificate ID    : {cert.certificate_id}")
    print(f"   - Overall Status    : {cert.overall_verdict.value}")
    print(f"   - Input SHA-256 Hash: {cert.input_hash}")
    print(f"   - Total Latency     : {cert.execution_time_seconds:.3f}s")
    print(f"   - Avg Claim Latency : {cert.summary.avg_time_per_claim_ms:.1f}ms")
    print(f"   - Conflicts Detected: {cert.summary.conflicts_detected}")

    print("\n4. Claim-by-Claim Detailed Analysis:")
    for c in cert.claims:
        print("-" * 75)
        conflict_tag = " [⚠️ CONFLICT DETECTED]" if c.conflict_detected else ""
        print(f"[{c.claim_id}] {c.verdict.value}{conflict_tag}: \"{c.claim_text}\"")
        print(f"     Rationale  : {c.explanation}")
        print(f"     Latency    : {c.processing_time_ms:.1f}ms")

        if c.primary_evidence:
            pe = c.primary_evidence
            print(f"     Primary Ev : \"{pe.text}\"")
            print(f"                  Source: {pe.source} | Tier: {pe.authority_level} (wt: {pe.authority_weight:.2f})")
            print(f"                  Cosine Sim: {pe.similarity:.3f} -> Ranking Score: {pe.ranking_score:.3f}")

        if c.conflicting_evidence:
            print(f"     Conflicting Evidence ({len(c.conflicting_evidence)} passage(s)):")
            for ce in c.conflicting_evidence:
                print(f"       • \"{ce.text}\"")
                print(f"         Source: {ce.source} | Tier: {ce.authority_level} | Cosine Sim: {ce.similarity:.3f}")

    print("=" * 75)
    print("✅ Advanced Verification Scenario Completed Successfully!")
    print("=" * 75)


if __name__ == "__main__":
    main()
