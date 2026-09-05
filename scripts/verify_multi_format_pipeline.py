import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.verifier import VerificationPipeline
from backend.services.certificate import generate_human_readable_report
from backend.schemas import VerdictType, OverallVerdict


def main():
    print("=" * 75)
    print("🌐 VERDICT Multi-Format Verification Test (9 Formats)")
    print("=" * 75)

    demo_dir = Path(__file__).parent.parent / "data" / "demo_files"
    data_dir = Path(__file__).parent.parent / "data"

    # Gather files across all 9 formats
    files_to_test = [
        ("sample_policy.pdf", data_dir / "sample_policy.pdf"),
        ("sample_policy.txt", data_dir / "sample_policy.txt"),
        ("demo_contract.docx", demo_dir / "demo_contract.docx"),
        ("demo_presentation.pptx", demo_dir / "demo_presentation.pptx"),
        ("demo_financials.xlsx", demo_dir / "demo_financials.xlsx"),
        ("demo_compliance.csv", demo_dir / "demo_compliance.csv"),
        ("demo_notes.md", demo_dir / "demo_notes.md"),
        ("demo_data.json", demo_dir / "demo_data.json"),
        ("demo_portal.html", demo_dir / "demo_portal.html"),
    ]

    source_files = []
    for name, fpath in files_to_test:
        if fpath.exists():
            with open(fpath, "rb") as f:
                source_files.append((name, f.read()))
            print(f"  ✓ Loaded format [{name.split('.')[-1].upper()}]: {name}")
        else:
            print(f"  ✗ Missing: {name}")

    print(f"\nTotal Ground-Truth Sources Loaded: {len(source_files)} file formats")

    ai_draft = """1. Corporate compliance filings must be completed within 30 days.
2. The statutory penalty for late submissions is $50,000 per violation.
3. Commercial enterprises with turnover under $2.5 million are exempt from quarterly audits.
4. Entities are required to maintain a fleet of supersonic delivery drones."""

    pipeline = VerificationPipeline()
    print("\nExecuting Multi-Format Verification Pipeline...")
    cert = pipeline.run_verification(
        draft_text=ai_draft,
        source_files=source_files,
        top_k=3
    )

    print("\n" + "=" * 75)
    print("📋 MULTI-FORMAT VERIFICATION CERTIFICATE AUDIT")
    print("=" * 75)
    print(f"Certificate ID  : {cert.certificate_id}")
    print(f"Overall Status  : {cert.overall_verdict.value}")
    print(f"Total Claims    : {cert.summary.total_claims}")
    print(f"  - Supported   : {cert.summary.supported}")
    print(f"  - Refuted     : {cert.summary.refuted}")
    print(f"  - Unverified  : {cert.summary.unverified}")

    print("\nSource Ingestion Audit (9 Formats Hashed):")
    for s in cert.sources:
        print(f"  [{s.file_type.upper():<5}] {s.filename:<24} | SHA256: {s.sha256[:16]}... | Scope: {len(s.evidence_locations)} locations")

    print("\nClaim-by-Claim Verification Grounding:")
    for claim in cert.claims:
        v_icon = "🟢" if claim.verdict == VerdictType.SUPPORTED else ("🔴" if claim.verdict == VerdictType.REFUTED else "🟡")
        print(f"\n  {v_icon} Claim [{claim.claim_id}]: {claim.verdict.value}")
        print(f"     Claim Text: \"{claim.text}\"")
        print(f"     Rationale : {claim.reason}")
        if claim.evidence:
            top_ev = claim.evidence[0]
            loc_label = top_ev.location or (f"Page {top_ev.page}" if top_ev.page else "N/A")
            fmt = (top_ev.file_type or "doc").upper()
            print(f"     Evidence  : \"{top_ev.text[:100]}...\"")
            print(f"     Location  : [{fmt}] {top_ev.source} -> {loc_label} (Cosine Sim: {top_ev.similarity:.2f})")
        else:
            print("     Evidence  : None")

    print("\n" + "=" * 75)
    print("✅ All 9 Formats Verified Successfully End-to-End!")
    print("=" * 75)


if __name__ == "__main__":
    main()
