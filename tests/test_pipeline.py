import pytest
from backend.services.verifier import VerificationPipeline
from backend.schemas import VerdictType, OverallVerdict
from backend.services.certificate import determine_overall_verdict, VerificationSummary


SOURCE_TEXT = """CORPORATE COMPLIANCE RULES:
Rule 1: Annual financial reports must be submitted within 30 days of the fiscal year end.
Rule 2: Submissions must be delivered through the electronic portal.
Rule 3: Late filing penalty is $15,000 per violation.
"""


def test_pipeline_end_to_end():
    pipeline = VerificationPipeline()

    draft = """1. Annual financial reports must be submitted within 30 days of fiscal year end.
2. Late filing penalty is $90,000 per violation.
3. Entities must install biometric employee access systems."""

    cert = pipeline.run_verification(
        draft_text=draft,
        source_files=[("compliance_rules.txt", SOURCE_TEXT.encode("utf-8"))],
        top_k=3
    )

    assert cert.certificate_id.startswith("vc_")
    assert len(cert.sources) == 1
    assert cert.sources[0].filename == "compliance_rules.txt"
    assert len(cert.sources[0].sha256) == 64
    assert len(cert.claims) >= 3

    # Check that individual claims have expected verdicts
    verdicts = [c.verdict for c in cert.claims]
    assert VerdictType.SUPPORTED in verdicts, "Expected at least one SUPPORTED claim"
    assert VerdictType.REFUTED in verdicts, "Expected at least one REFUTED claim ($90,000 vs $15,000)"
    assert VerdictType.UNVERIFIED in verdicts, "Expected at least one UNVERIFIED claim (biometric systems)"

    # Since at least one claim is REFUTED and one UNVERIFIED, overall verdict must be REVIEW_REQUIRED
    assert cert.overall_verdict == OverallVerdict.REVIEW_REQUIRED


def test_pipeline_with_real_pdf():
    from pathlib import Path
    pdf_path = Path(__file__).parent.parent / "data" / "sample_policy.pdf"
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        pipeline = VerificationPipeline()
        draft = "Corporate filings are due within 30 days. Late filing penalty is $50,000."
        cert = pipeline.run_verification(
            draft_text=draft,
            source_files=[("sample_policy.pdf", pdf_bytes)],
            top_k=3
        )
        assert len(cert.claims) >= 2
        assert cert.sources[0].file_type == "pdf"
        assert cert.sources[0].page_count >= 1
        verdicts = [c.verdict for c in cert.claims]
        assert VerdictType.SUPPORTED in verdicts
        assert VerdictType.REFUTED in verdicts


def test_overall_verdict_conservative_logic():
    # All supported -> VERIFIED
    s1 = VerificationSummary(total_claims=3, supported=3, refuted=0, unverified=0)
    assert determine_overall_verdict(s1) == OverallVerdict.VERIFIED

    # Any refuted -> REVIEW_REQUIRED
    s2 = VerificationSummary(total_claims=3, supported=2, refuted=1, unverified=0)
    assert determine_overall_verdict(s2) == OverallVerdict.REVIEW_REQUIRED

    # Any unverified -> REVIEW_REQUIRED
    s3 = VerificationSummary(total_claims=3, supported=2, refuted=0, unverified=1)
    assert determine_overall_verdict(s3) == OverallVerdict.REVIEW_REQUIRED

    # Empty -> REVIEW_REQUIRED
    s4 = VerificationSummary(total_claims=0, supported=0, refuted=0, unverified=0)
    assert determine_overall_verdict(s4) == OverallVerdict.REVIEW_REQUIRED
