import pytest
from backend.schemas import VerdictType, OverallVerdict
from backend.services.verifier import VerificationPipeline
from backend.services.claim_decomposer import decompose_draft


def test_matrix_a_two_docs_two_matching_claims():
    """
    Test A: Two source documents + two claims from each document.
    Doc A: Information Security & Access Control Policy (account locked after 5 failed login attempts).
    Doc B: Company Leave & Attendance Policy (20 days of paid annual leave per calendar year).
    Claim 1: An account will be temporarily locked after 5 consecutive failed login attempts.
    Claim 2: Full-time employees are entitled to 20 days of paid annual leave per calendar year.

    Expected:
    - Total claims: 2
    - Supported: 2
    - Refuted: 0
    - Unverified: 0
    - Conflicts: 0
    - Overall verdict: VERIFIED
    - Claim 1 primary evidence from Doc A
    - Claim 2 primary evidence from Doc B
    """
    doc_a = (
        "Information Security & Access Control Policy\n"
        "An account will be temporarily locked after 5 consecutive failed login attempts."
    )
    doc_b = (
        "Company Leave & Attendance Policy\n"
        "Full-time employees are entitled to 20 days of paid annual leave per calendar year."
    )

    draft = """CLAIM 1:
"An account will be temporarily locked after 5 consecutive failed login attempts."

CLAIM 2:
"Full-time employees are entitled to 20 days of paid annual leave per calendar year."
"""
    pipeline = VerificationPipeline()
    cert = pipeline.run_verification(
        draft_text=draft,
        source_files=[
            ("SecPolicy.txt", doc_a.encode("utf-8")),
            ("LeavePolicy.txt", doc_b.encode("utf-8"))
        ],
        top_k=3
    )

    assert cert.summary.total_claims == 2
    assert cert.summary.supported == 2
    assert cert.summary.refuted == 0
    assert cert.summary.unverified == 0
    assert cert.summary.conflicts_detected == 0
    assert cert.overall_verdict == OverallVerdict.VERIFIED

    # Check Claim 1
    c1 = cert.claims[0]
    assert c1.verdict == VerdictType.SUPPORTED
    assert c1.confidence >= 0.90
    assert c1.conflict_detected is False
    assert c1.primary_evidence is not None
    assert c1.primary_evidence.source == "SecPolicy.txt"

    # Check Claim 2
    c2 = cert.claims[1]
    assert c2.verdict == VerdictType.SUPPORTED
    assert c2.confidence >= 0.90
    assert c2.conflict_detected is False
    assert c2.primary_evidence is not None
    assert c2.primary_evidence.source == "LeavePolicy.txt"


def test_matrix_b_two_docs_one_match_one_numeric_mismatch():
    """
    Test B: Two source documents + 1 matching claim + 1 numeric mismatch claim.
    Claim 1: 5 consecutive failed login attempts (matches Doc A).
    Claim 2: 25 paid annual leave days (Doc B says 20 days).

    Expected:
    - Total claims: 2
    - Supported: 1
    - Refuted: 1
    - Unverified: 0
    - Conflicts: 0
    - Overall verdict: REVIEW_REQUIRED
    - Claim 1: SUPPORTED (Doc A)
    - Claim 2: REFUTED (Doc B, mentions 25 and 20)
    """
    doc_a = (
        "Information Security & Access Control Policy\n"
        "An account will be temporarily locked after 5 consecutive failed login attempts."
    )
    doc_b = (
        "Company Leave & Attendance Policy\n"
        "Full-time employees are entitled to 20 days of paid annual leave per calendar year."
    )

    draft = """CLAIM 1:
"An account will be temporarily locked after 5 consecutive failed login attempts."

CLAIM 2:
"Full-time employees are entitled to 25 paid annual leave days per calendar year."
"""
    pipeline = VerificationPipeline()
    cert = pipeline.run_verification(
        draft_text=draft,
        source_files=[
            ("SecPolicy.txt", doc_a.encode("utf-8")),
            ("LeavePolicy.txt", doc_b.encode("utf-8"))
        ],
        top_k=3
    )

    assert cert.summary.total_claims == 2
    assert cert.summary.supported == 1
    assert cert.summary.refuted == 1
    assert cert.summary.unverified == 0
    assert cert.summary.conflicts_detected == 0
    assert cert.overall_verdict == OverallVerdict.REVIEW_REQUIRED

    c1 = cert.claims[0]
    assert c1.verdict == VerdictType.SUPPORTED
    assert c1.primary_evidence.source == "SecPolicy.txt"

    c2 = cert.claims[1]
    assert c2.verdict == VerdictType.REFUTED
    assert c2.primary_evidence.source == "LeavePolicy.txt"
    assert "25" in c2.reason
    assert "20" in c2.reason


def test_matrix_c_independent_source_attribution():
    """
    Test C: Two docs from distinct domains; verify claims independently retrieve and attribute evidence.
    """
    doc_financial = (
        "Q4 Financial Statement\n"
        "The corporate operating budget for fiscal year 2024 was set at $50 million."
    )
    doc_hr = (
        "Human Resources Code of Conduct\n"
        "Standard working hours are strictly 40 hours per calendar week."
    )

    draft = (
        "The corporate operating budget for fiscal year 2024 was set at $50 million. "
        "Standard working hours are strictly 40 hours per calendar week."
    )
    pipeline = VerificationPipeline()
    cert = pipeline.run_verification(
        draft_text=draft,
        source_files=[
            ("financials.txt", doc_financial.encode("utf-8")),
            ("hr_policy.txt", doc_hr.encode("utf-8"))
        ],
        top_k=3
    )

    assert cert.summary.total_claims == 2
    assert cert.summary.supported == 2
    assert cert.overall_verdict == OverallVerdict.VERIFIED

    assert cert.claims[0].primary_evidence.source == "financials.txt"
    assert cert.claims[1].primary_evidence.source == "hr_policy.txt"


def test_matrix_d_unrelated_subjects_zero_false_conflicts():
    """
    Test D: Two documents on different subjects sharing generic words ("employee", "days").
    Doc A: Employees must report security breaches within 5 days.
    Doc B: Full-time employees are entitled to 20 days of paid annual leave per calendar year.
    Claim: Employees must report security breaches within 5 days.

    Expected:
    - Zero false conflicts (Doc B must NOT be treated as contradicting Doc A).
    - Claim is SUPPORTED.
    - Overall verdict: VERIFIED.
    """
    doc_a = "Information Security Policy\nEmployees must report security breaches within 5 days."
    doc_b = "Company Leave Policy\nFull-time employees are entitled to 20 days of paid annual leave per calendar year."

    pipeline = VerificationPipeline()
    cert = pipeline.run_verification(
        draft_text="Employees must report security breaches within 5 days.",
        source_files=[
            ("SecPolicy.txt", doc_a.encode("utf-8")),
            ("LeavePolicy.txt", doc_b.encode("utf-8"))
        ],
        top_k=3
    )

    assert cert.summary.total_claims == 1
    assert cert.summary.supported == 1
    assert cert.summary.conflicts_detected == 0
    assert cert.claims[0].conflict_detected is False
    assert cert.overall_verdict == OverallVerdict.VERIFIED


def test_matrix_e_genuine_cross_source_contradiction():
    """
    Test E: Two documents with a genuine contradiction on the exact same policy metric.
    Doc A (v1): Accounts are locked after 5 consecutive failed login attempts.
    Doc B (v2): Accounts are locked after 3 consecutive failed login attempts.
    Claim: Accounts are locked after 5 consecutive failed login attempts.

    Expected:
    - Claim-level cross-source conflict detected.
    - Overall verdict: REVIEW_REQUIRED.
    - Conflict sources identify both documents.
    """
    doc_a = "Information Security Policy v1\nAccounts are locked after 5 consecutive failed login attempts."
    doc_b = "Information Security Policy v2\nAccounts are locked after 3 consecutive failed login attempts."

    pipeline = VerificationPipeline()
    cert = pipeline.run_verification(
        draft_text="Accounts are locked after 5 consecutive failed login attempts.",
        source_files=[
            ("SecPolicy_v1.txt", doc_a.encode("utf-8")),
            ("SecPolicy_v2.txt", doc_b.encode("utf-8"))
        ],
        top_k=3
    )

    assert cert.summary.total_claims == 1
    assert cert.claims[0].conflict_detected is True
    assert cert.summary.conflicts_detected == 1
    assert cert.overall_verdict == OverallVerdict.REVIEW_REQUIRED
    assert len(cert.claims[0].conflicting_evidence) >= 1
    assert "SecPolicy_v2.txt" in cert.claims[0].reason


def test_claim_decomposer_formatting_variations():
    """
    Tests claim decomposer with various draft formatting:
    - Inline 'CLAIM 1: ...'
    - Multiline 'CLAIM 1:\n"..."'
    - Bullet points '1. "..."'
    """
    draft = """CLAIM 1:
"An account will be temporarily locked after 5 consecutive failed login attempts."

CLAIM 2:
"Full-time employees are entitled to 20 days of paid annual leave per calendar year."
"""
    claims = decompose_draft(draft)
    assert len(claims) == 2
    assert claims[0].claim_id == "C001"
    assert claims[1].claim_id == "C002"
    assert not claims[0].text.startswith("CLAIM")
    assert not claims[0].text.startswith('"')
    assert not claims[1].text.startswith("CLAIM")
    assert not claims[1].text.startswith('"')
    assert "5 consecutive failed login attempts" in claims[0].text
    assert "20 days of paid annual leave" in claims[1].text


def test_explicit_claims_list_input_pipeline():
    """
    Test 7 (Requirement): Submits TWO claims against TWO unrelated documents using explicit claims list.
    Asserts:
    - len(results) == 2
    - C001 exists
    - C002 exists
    - Both have independent verdicts and independent primary sources.
    """
    doc_a = (
        "Information Security & Access Control Policy\n"
        "An account will be temporarily locked after 5 consecutive failed login attempts."
    )
    doc_b = (
        "Company Leave & Attendance Policy\n"
        "Full-time employees are entitled to 20 days of paid annual leave per calendar year."
    )

    claims_input = [
        "An account will be temporarily locked after 5 consecutive failed login attempts.",
        "Full-time employees are entitled to 20 days of paid annual leave per calendar year."
    ]

    pipeline = VerificationPipeline()
    cert = pipeline.run_verification(
        claims=claims_input,
        source_files=[
            ("SecPolicy.txt", doc_a.encode("utf-8")),
            ("LeavePolicy.txt", doc_b.encode("utf-8"))
        ],
        top_k=3
    )

    # Asserts
    assert len(cert.claims) == 2
    assert cert.summary.total_claims == 2
    assert cert.claims[0].claim_id == "C001"
    assert cert.claims[1].claim_id == "C002"

    assert cert.claims[0].verdict == VerdictType.SUPPORTED
    assert cert.claims[0].primary_evidence is not None
    assert cert.claims[0].primary_evidence.source == "SecPolicy.txt"

    assert cert.claims[1].verdict == VerdictType.SUPPORTED
    assert cert.claims[1].primary_evidence is not None
    assert cert.claims[1].primary_evidence.source == "LeavePolicy.txt"

    assert cert.summary.supported == 2
    assert cert.summary.refuted == 0
    assert cert.summary.unverified == 0
    assert cert.summary.conflicts_detected == 0
    assert cert.overall_verdict == OverallVerdict.VERIFIED


def test_two_claims_one_correct_one_numerically_incorrect():
    """
    Test 8 (Requirement):
    Claim 1 correct -> SUPPORTED
    Claim 2 numerically incorrect -> REFUTED
    Expected:
    Total Claims: 2
    Supported: 1
    Refuted: 1
    Unverified: 0
    """
    doc_a = (
        "Information Security & Access Control Policy\n"
        "An account will be temporarily locked after 5 consecutive failed login attempts."
    )
    doc_b = (
        "Company Leave & Attendance Policy\n"
        "Full-time employees are entitled to 20 days of paid annual leave per calendar year."
    )

    claims_input = [
        "An account will be temporarily locked after 5 consecutive failed login attempts.",
        "Full-time employees are entitled to 25 paid annual leave days per calendar year."
    ]

    pipeline = VerificationPipeline()
    cert = pipeline.run_verification(
        claims=claims_input,
        source_files=[
            ("SecPolicy.txt", doc_a.encode("utf-8")),
            ("LeavePolicy.txt", doc_b.encode("utf-8"))
        ],
        top_k=3
    )

    assert cert.summary.total_claims == 2
    assert cert.summary.supported == 1
    assert cert.summary.refuted == 1
    assert cert.summary.unverified == 0
    assert cert.summary.conflicts_detected == 0
    assert cert.overall_verdict == OverallVerdict.REVIEW_REQUIRED

    assert cert.claims[0].claim_id == "C001"
    assert cert.claims[0].verdict == VerdictType.SUPPORTED
    assert cert.claims[0].primary_evidence.source == "SecPolicy.txt"

    assert cert.claims[1].claim_id == "C002"
    assert cert.claims[1].verdict == VerdictType.REFUTED
    assert cert.claims[1].primary_evidence.source == "LeavePolicy.txt"
    assert "25" in cert.claims[1].reason
    assert "20" in cert.claims[1].reason


def test_fastapi_verify_endpoint_claims_parameter():
    """
    Verifies that FastAPI POST /verify correctly accepts the 'claims' Form field
    with a JSON array of claims and executes multi-claim verification end-to-end.
    """
    import io
    import json
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    doc_a = "Information Security & Access Control Policy\nAn account will be temporarily locked after 5 consecutive failed login attempts."
    doc_b = "Company Leave & Attendance Policy\nFull-time employees are entitled to 20 days of paid annual leave per calendar year."

    claims_list = [
        "An account will be temporarily locked after 5 consecutive failed login attempts.",
        "Full-time employees are entitled to 20 days of paid annual leave per calendar year."
    ]

    files = [
        ("source_files", ("SecPolicy.txt", io.BytesIO(doc_a.encode("utf-8")), "text/plain")),
        ("source_files", ("LeavePolicy.txt", io.BytesIO(doc_b.encode("utf-8")), "text/plain")),
    ]

    response = client.post("/verify", data={"claims": json.dumps(claims_list)}, files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    cert = data["certificate"]

    assert cert["summary"]["total_claims"] == 2
    assert cert["summary"]["supported"] == 2
    assert cert["summary"]["refuted"] == 0
    assert cert["summary"]["unverified"] == 0
    assert cert["overall_verdict"] == "VERIFIED"

    assert len(cert["claims"]) == 2
    assert cert["claims"][0]["claim_id"] == "C001"
    assert cert["claims"][0]["verdict"] == "SUPPORTED"
    assert cert["claims"][1]["claim_id"] == "C002"
    assert cert["claims"][1]["verdict"] == "SUPPORTED"


def test_json_and_syntax_variations_decomposition():
    """
    Tests decomposition of JSON array, Python assignment, semicolon-separated,
    and comma-separated quoted claims.
    """
    from backend.services.claim_decomposer import decompose_draft

    # 1. JSON list string
    json_draft = '["Claim one is here.", "Claim two is here."]'
    c_json = decompose_draft(json_draft)
    assert len(c_json) == 2
    assert c_json[0].claim_id == "C001"
    assert c_json[1].claim_id == "C002"

    # 2. Python variable assignment
    py_draft = 'claims = [\n  "First claim statement.",\n  "Second claim statement."\n]'
    c_py = decompose_draft(py_draft)
    assert len(c_py) == 2
    assert c_py[0].claim_id == "C001"
    assert c_py[1].claim_id == "C002"

    # 3. Semicolon separated
    semi_draft = "First policy requires submission within 30 days; Second policy states full compliance is mandatory."
    c_semi = decompose_draft(semi_draft)
    assert len(c_semi) == 2
    assert c_semi[0].claim_id == "C001"
    assert c_semi[1].claim_id == "C002"


def test_regression_multi_source_conflict_20_vs_25_days():
    """
    Regression test:
    Source 1 (Sample.txt): 20 days annual leave.
    Source 2 (VERDICT_Test_Source_2026.txt): 25 days annual leave.
    Claim: 25 days annual leave.
    Expected:
    - Total claims: 1
    - Supported: 1
    - Refuted: 0
    - Conflicts: 1
    - Overall: REVIEW_REQUIRED
    - Both sources identified (supporting = Source 2, conflicting = Source 1).
    """
    pipeline = VerificationPipeline()
    doc_1 = "Company Leave & Attendance Policy\nFull-time employees are entitled to 20 days of paid annual leave per calendar year."
    doc_2 = "Company Leave Policy 2026\nFull-time employees are entitled to 25 days of paid annual leave per calendar year."
    claim = "Full-time employees are entitled to 25 paid annual leave days per calendar year."

    cert = pipeline.run_verification(
        draft_text=claim,
        source_files=[
            ("Sample.txt", doc_1.encode("utf-8")),
            ("VERDICT_Test_Source_2026.txt", doc_2.encode("utf-8"))
        ],
        top_k=3
    )

    assert cert.summary.total_claims == 1
    assert cert.summary.supported == 1
    assert cert.summary.refuted == 0
    assert cert.summary.unverified == 0
    assert cert.summary.conflicts_detected == 1
    assert cert.overall_verdict == OverallVerdict.REVIEW_REQUIRED

    c = cert.claims[0]
    assert c.verdict == VerdictType.SUPPORTED
    assert c.conflict_detected is True
    assert c.primary_evidence is not None
    assert c.primary_evidence.source == "VERDICT_Test_Source_2026.txt"
    assert len(c.conflicting_evidence) >= 1
    assert any(ce.source == "Sample.txt" for ce in c.conflicting_evidence)


def test_regression_multi_source_conflict_order_independence():
    """
    Tests that conflict detection and summary metrics are invariant to upload/retrieval order.
    Uploads supporting source first, then refuting source.
    """
    pipeline = VerificationPipeline()
    doc_1 = "Company Leave & Attendance Policy\nFull-time employees are entitled to 20 days of paid annual leave per calendar year."
    doc_2 = "Company Leave Policy 2026\nFull-time employees are entitled to 25 days of paid annual leave per calendar year."
    claim = "Full-time employees are entitled to 25 paid annual leave days per calendar year."

    cert = pipeline.run_verification(
        draft_text=claim,
        source_files=[
            ("VERDICT_Test_Source_2026.txt", doc_2.encode("utf-8")),
            ("Sample.txt", doc_1.encode("utf-8"))
        ],
        top_k=3
    )

    assert cert.summary.total_claims == 1
    assert cert.summary.supported == 1
    assert cert.summary.refuted == 0
    assert cert.summary.unverified == 0
    assert cert.summary.conflicts_detected == 1
    assert cert.overall_verdict == OverallVerdict.REVIEW_REQUIRED
    assert cert.claims[0].conflict_detected is True


def test_regression_two_sources_both_supporting():
    """
    Tests that when two uploaded sources both support the claim:
    - Total claims: 1
    - Supported: 1
    - Refuted: 0
    - Conflicts: 0
    - Overall: VERIFIED
    """
    pipeline = VerificationPipeline()
    doc_a = "Policy North America\nFull-time employees are entitled to 25 days of paid annual leave per calendar year."
    doc_b = "Policy Europe 2026\nFull-time employees are entitled to 25 days of paid annual leave per calendar year."
    claim = "Full-time employees are entitled to 25 paid annual leave days per calendar year."

    cert = pipeline.run_verification(
        draft_text=claim,
        source_files=[
            ("Policy_NA.txt", doc_a.encode("utf-8")),
            ("Policy_EU.txt", doc_b.encode("utf-8"))
        ],
        top_k=3
    )

    assert cert.summary.total_claims == 1
    assert cert.summary.supported == 1
    assert cert.summary.refuted == 0
    assert cert.summary.unverified == 0
    assert cert.summary.conflicts_detected == 0
    assert cert.overall_verdict == OverallVerdict.VERIFIED
    assert cert.claims[0].conflict_detected is False


def test_regression_two_sources_both_refuting():
    """
    Tests that when two uploaded sources both contradict the claim:
    - Total claims: 1
    - Supported: 0
    - Refuted: 1
    - Conflicts: 0 (unanimous refutation, no inter-source disagreement)
    - Overall: REVIEW_REQUIRED
    """
    pipeline = VerificationPipeline()
    doc_a = "Policy v1\nFull-time employees are entitled to 20 days of paid annual leave per calendar year."
    doc_b = "Policy v2\nFull-time employees are entitled to 18 days of paid annual leave per calendar year."
    claim = "Full-time employees are entitled to 25 paid annual leave days per calendar year."

    cert = pipeline.run_verification(
        draft_text=claim,
        source_files=[
            ("Policy_v1.txt", doc_a.encode("utf-8")),
            ("Policy_v2.txt", doc_b.encode("utf-8"))
        ],
        top_k=3
    )

    assert cert.summary.total_claims == 1
    assert cert.summary.supported == 0
    assert cert.summary.refuted == 1
    assert cert.summary.unverified == 0
    assert cert.summary.conflicts_detected == 0
    assert cert.overall_verdict == OverallVerdict.REVIEW_REQUIRED
    assert cert.claims[0].verdict == VerdictType.REFUTED
    assert cert.claims[0].conflict_detected is False


def test_regression_unrelated_second_doc_no_false_conflict():
    """
    Tests that unrelated evidence in a second uploaded document does NOT create a false conflict.
    """
    pipeline = VerificationPipeline()
    doc_hr = "Company Leave Policy 2026\nFull-time employees are entitled to 25 days of paid annual leave per calendar year."
    doc_fin = "Q4 Financial Operations\nThe operating budget for fiscal year 2024 was set at $50 million."
    claim = "Full-time employees are entitled to 25 paid annual leave days per calendar year."

    cert = pipeline.run_verification(
        draft_text=claim,
        source_files=[
            ("LeavePolicy_2026.txt", doc_hr.encode("utf-8")),
            ("Financials_2024.txt", doc_fin.encode("utf-8"))
        ],
        top_k=3
    )

    assert cert.summary.total_claims == 1
    assert cert.summary.supported == 1
    assert cert.summary.refuted == 0
    assert cert.summary.unverified == 0
    assert cert.summary.conflicts_detected == 0
    assert cert.overall_verdict == OverallVerdict.VERIFIED
    assert cert.claims[0].conflict_detected is False
    assert cert.claims[0].primary_evidence.source == "LeavePolicy_2026.txt"

