import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.verifier import VerificationPipeline
from backend.services.certificate import generate_human_readable_report
from backend.schemas import VerdictType, OverallVerdict


def main():
    print("=" * 70)
    print("🚀 Running VERDICT End-to-End Verification Pipeline")
    print("=" * 70)

    data_dir = Path(__file__).parent.parent / "data"
    pdf_path = data_dir / "sample_policy.pdf"
    txt_path = data_dir / "sample_policy.txt"

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    with open(txt_path, "rb") as f:
        txt_bytes = f.read()

    ai_draft = """1. All commercial entities must complete their corporate filings within 30 days of the fiscal year close.
2. The statutory penalty for late filing is $50,000 per violation.
3. Regulated organizations must maintain an on-premise quantum server cluster."""

    pipeline = VerificationPipeline()
    print("\n1. Ingesting & Hashing Source Documents...")
    print(f"   - PDF: {pdf_path.name} ({len(pdf_bytes)} bytes)")
    print(f"   - TXT: {txt_path.name} ({len(txt_bytes)} bytes)")

    print("\n2. Executing Claim Verification Pipeline...")
    certificate = pipeline.run_verification(
        draft_text=ai_draft,
        source_files=[
            ("sample_policy.pdf", pdf_bytes),
            ("sample_policy.txt", txt_bytes)
        ],
        top_k=3
    )

    print("\n3. Verification Results Summary:")
    print(f"   - Certificate ID : {certificate.certificate_id}")
    print(f"   - Overall Verdict: {certificate.overall_verdict.value}")
    print(f"   - Total Claims   : {certificate.summary.total_claims}")
    print(f"   - Supported      : {certificate.summary.supported}")
    print(f"   - Refuted        : {certificate.summary.refuted}")
    print(f"   - Unverified     : {certificate.summary.unverified}")

    print("\n4. Cryptographic Source Audit:")
    for src in certificate.sources:
        print(f"   - [{src.filename}] SHA-256: {src.sha256} (Pages: {src.page_count})")

    print("\n5. Claim-by-Claim Verification Audit:")
    for claim in certificate.claims:
        v_icon = "🟢" if claim.verdict == VerdictType.SUPPORTED else ("🔴" if claim.verdict == VerdictType.REFUTED else "🟡")
        print(f"\n   {v_icon} Claim [{claim.claim_id}]: {claim.verdict.value}")
        print(f"      Text: \"{claim.text}\"")
        print(f"      Reason: {claim.reason}")
        if claim.evidence:
            top_ev = claim.evidence[0]
            print(f"      Grounding: \"{top_ev.text}\" (Source: {top_ev.source}, Page: {top_ev.page}, Sim: {top_ev.similarity:.2f})")
        else:
            print("      Grounding: None")

    print("\n6. Exporting Human-Readable Verification Report...")
    report_text = generate_human_readable_report(certificate)
    report_path = data_dir / f"{certificate.certificate_id}_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"   - Saved report to: {report_path}")

    print("\n" + "=" * 70)
    print("✅ End-to-End Verification Pipeline Successfully Verified!")
    print("=" * 70)


if __name__ == "__main__":
    main()
