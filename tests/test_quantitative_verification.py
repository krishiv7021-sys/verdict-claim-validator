import io
import pytest
import docx

from backend.schemas import VerdictType, OverallVerdict, EvidenceSpan
from backend.services.verifier import VerificationPipeline
from backend.services.entailment import (
    HeuristicNLIEntailmentEngine,
    check_quantitative_contradiction,
    extract_quantitative_entities,
    is_quantitative_claim_grounded
)


def create_sample_docx(text_paragraphs):
    """Helper to create an in-memory DOCX file with specified paragraphs."""
    doc = docx.Document()
    for p in text_paragraphs:
        doc.add_paragraph(p)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_exact_annual_leave_20_vs_25_pipeline_case():
    """
    CRITICAL REGRESSION TEST: Exact user-reported bug.
    Evidence document: Company Leave & Attendance Policy (Sample.docx)
    Evidence: "Full-time employees are entitled to 20 days of paid annual leave per calendar year."
    Claim: "Full-time employees are entitled to 25 paid annual leave days per calendar year."

    Expected result:
    - Total Claims: 1
    - Supported: 0
    - Refuted: 1
    - Unverified: 0
    - Source Conflicts: 0
    - Overall status: REVIEW_REQUIRED (NOT VERIFIED)
    - Rationale explains 25 vs 20 days and NEVER claims evidence confirms claim.
    """
    docx_bytes = create_sample_docx([
        "Company Leave & Attendance Policy",
        "Full-time employees are entitled to 20 days of paid annual leave per calendar year."
    ])

    pipeline = VerificationPipeline()
    claim_draft = "Full-time employees are entitled to 25 paid annual leave days per calendar year."

    cert = pipeline.run_verification(
        draft_text=claim_draft,
        source_files=[("Sample.docx", docx_bytes)],
        top_k=3
    )

    # Validate overall counts
    assert cert.summary.total_claims == 1
    assert cert.summary.supported == 0
    assert cert.summary.refuted == 1
    assert cert.summary.unverified == 0
    assert cert.summary.conflicts_detected == 0
    assert cert.claims[0].conflict_detected is False

    # Overall verdict must NOT be VERIFIED
    assert cert.overall_verdict != OverallVerdict.VERIFIED
    assert cert.overall_verdict == OverallVerdict.REVIEW_REQUIRED

    # Claim-level inspection
    claim_result = cert.claims[0]
    assert claim_result.verdict == VerdictType.REFUTED
    assert claim_result.confidence >= 0.90
    assert "25" in claim_result.reason
    assert "20" in claim_result.reason
    assert "confirms" not in claim_result.reason.lower()
    assert "Sample.docx" in claim_result.reason


def test_exact_annual_leave_20_vs_20_match_pipeline_case():
    """Exact match: 20 days in claim and 20 days in evidence should be SUPPORTED."""
    docx_bytes = create_sample_docx([
        "Company Leave & Attendance Policy",
        "Full-time employees are entitled to 20 days of paid annual leave per calendar year."
    ])

    pipeline = VerificationPipeline()
    claim_draft = "Full-time employees are entitled to 20 paid annual leave days per calendar year."

    cert = pipeline.run_verification(
        draft_text=claim_draft,
        source_files=[("Sample.docx", docx_bytes)],
        top_k=3
    )

    assert cert.summary.total_claims == 1
    assert cert.summary.supported == 1
    assert cert.summary.refuted == 0
    assert cert.summary.unverified == 0
    assert cert.overall_verdict == OverallVerdict.VERIFIED
    assert cert.claims[0].verdict == VerdictType.SUPPORTED


def test_count_entity_contradiction():
    """Different numeric count for the same entity context (5 vs 10 engineers)."""
    engine = HeuristicNLIEntailmentEngine()
    span = EvidenceSpan(
        source="team_structure.txt",
        text="The platform engineering department employs 5 senior infrastructure engineers.",
        similarity=0.90
    )
    claim = "The platform engineering department employs 10 senior infrastructure engineers."
    verdict, conf, reason = engine.verify_span(claim, span)

    assert verdict == VerdictType.REFUTED
    assert conf >= 0.90
    assert "10" in reason
    assert "5" in reason


def test_percentage_contradiction():
    """Percentage discrepancy on interest rate (15% vs 20%)."""
    engine = HeuristicNLIEntailmentEngine()
    span = EvidenceSpan(
        source="loan_terms.txt",
        text="The annual loan interest rate is fixed at 20% per annum.",
        similarity=0.92
    )
    claim = "The annual loan interest rate is fixed at 15% per annum."
    verdict, conf, reason = engine.verify_span(claim, span)

    assert verdict == VerdictType.REFUTED
    assert "15%" in reason
    assert "20%" in reason


def test_date_and_year_contradiction():
    """Date discrepancy (June 15 vs June 20) and Year discrepancy (2024 vs 2025)."""
    engine = HeuristicNLIEntailmentEngine()

    # Date test
    span_date = EvidenceSpan(
        source="calendar.txt",
        text="The mandatory statutory filing deadline is June 20.",
        similarity=0.88
    )
    verdict, conf, reason = engine.verify_span("The mandatory statutory filing deadline is June 15.", span_date)
    assert verdict == VerdictType.REFUTED
    assert "June 15" in reason
    assert "June 20" in reason

    # Year test
    span_year = EvidenceSpan(
        source="history.txt",
        text="The merger agreement between the parties was executed in 2025.",
        similarity=0.89
    )
    verdict_yr, conf_yr, reason_yr = engine.verify_span("The merger agreement between the parties was executed in 2024.", span_year)
    assert verdict_yr == VerdictType.REFUTED
    assert "2024" in reason_yr
    assert "2025" in reason_yr


def test_up_to_limit_contradiction_and_match():
    """'Up to' limit mismatch (up to 5 vs up to 2) and match (up to 2 vs up to 2)."""
    engine = HeuristicNLIEntailmentEngine()

    # Mismatch
    span = EvidenceSpan(
        source="remote_policy.txt",
        text="Remote work is allowed up to 2 days per week for engineering teams.",
        similarity=0.91
    )
    claim_diff = "Remote work is allowed up to 5 days per week for engineering teams."
    v_diff, c_diff, r_diff = engine.verify_span(claim_diff, span)
    assert v_diff == VerdictType.REFUTED
    assert "5" in r_diff
    assert "2" in r_diff

    # Match
    claim_same = "Remote work is allowed up to 2 days per week for engineering teams."
    v_same, c_same, r_same = engine.verify_span(claim_same, span)
    assert v_same == VerdictType.SUPPORTED


def test_at_least_modifier_contradiction():
    """'At least' requirement mismatch (at least 5 years vs at least 3 years)."""
    engine = HeuristicNLIEntailmentEngine()
    span = EvidenceSpan(
        source="job_req.txt",
        text="Qualified candidates must possess at least 3 years of production experience.",
        similarity=0.90
    )
    claim = "Qualified candidates must possess at least 5 years of production experience."
    v, c, r = engine.verify_span(claim, span)
    assert v == VerdictType.REFUTED
    assert "at least" in r.lower()
    assert "5" in r
    assert "3" in r


def test_more_than_vs_less_than_boundary_contradiction():
    """Opposite direction boundaries ('more than' vs 'less than')."""
    engine = HeuristicNLIEntailmentEngine()
    span = EvidenceSpan(
        source="fund_charter.txt",
        text="The venture fund requires less than $10 million in initial subscriber commitments.",
        similarity=0.88
    )
    claim = "The venture fund requires more than $10 million in initial subscriber commitments."
    v, c, r = engine.verify_span(claim, span)
    assert v == VerdictType.REFUTED
    assert "boundary contradiction" in r.lower() or "contradiction" in r.lower()


def test_unrelated_numbers_different_context_no_false_refutation():
    """
    Ensure numbers in different factual contexts within the same passage are NOT falsely compared.
    50 employees (count) and founded in 2015 (year) vs $10 million in 2022.
    """
    has_contra, reason = check_quantitative_contradiction(
        "The enterprise has 50 employees and was established in 2015.",
        "The enterprise has 50 employees and generated $10 million in 2022."
    )
    # The 2015 founding year should not be compared against the 2022 revenue year!
    assert has_contra is False
    assert reason is None


def test_ungrounded_quantitative_claim_is_unverified():
    """
    If a claim asserts a specific quantity not corroborated by the source passage,
    it must NOT be marked SUPPORTED despite semantic keyword overlap.
    """
    engine = HeuristicNLIEntailmentEngine()
    span = EvidenceSpan(
        source="annual_review.txt",
        text="The corporation experienced exceptional annual growth and expanding commercial operations.",
        similarity=0.75
    )
    claim = "The corporation experienced $50 million in annual growth."
    v, c, r = engine.verify_span(claim, span)

    assert v != VerdictType.SUPPORTED
    assert v == VerdictType.UNVERIFIED
    assert "quantitative" in r.lower() or "not fully establish" in r.lower()
