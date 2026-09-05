import pytest
from backend.services.verifier import VerificationPipeline
from backend.services.retriever import compute_ranking_score
from backend.schemas import (
    SourceAuthorityLevel,
    AUTHORITY_WEIGHTS,
    DEFAULT_AUTHORITY_LEVEL,
    DEFAULT_AUTHORITY_WEIGHT,
    OverallVerdict,
    VerdictType,
    normalize_authority_level,
)


def test_authority_normalization():
    level, weight = normalize_authority_level("STATUTORY / OFFICIAL")
    assert level == "STATUTORY / OFFICIAL"
    assert weight == 1.00

    level, weight = normalize_authority_level("statutory")
    assert level == "STATUTORY / OFFICIAL"
    assert weight == 1.00

    level, weight = normalize_authority_level("POLICY")
    assert level == "POLICY / REGULATION"
    assert weight == 0.97

    level, weight = normalize_authority_level("internal")
    assert level == "INTERNAL / ORGANIZATIONAL"
    assert weight == 0.94

    level, weight = normalize_authority_level("web")
    assert level == "REFERENCE / WEB"
    assert weight == 0.90

    # Default fallback
    level, weight = normalize_authority_level(None)
    assert level == DEFAULT_AUTHORITY_LEVEL
    assert weight == DEFAULT_AUTHORITY_WEIGHT


def test_authority_ranking_score_formula():
    # Formula: similarity * (0.75 + 0.25 * authority_weight)
    # 1. Statutory with similarity 0.60
    score_stat = compute_ranking_score(0.60, 1.00)
    assert score_stat == 0.60  # 0.60 * (0.75 + 0.25) = 0.60

    # 2. Reference with similarity 0.60
    score_ref = compute_ranking_score(0.60, 0.90)
    # 0.60 * (0.75 + 0.225) = 0.60 * 0.975 = 0.585
    assert score_ref == 0.585

    # 3. High relevance web source strictly outranks low relevance statutory source
    score_high_web = compute_ranking_score(0.85, 0.90)  # 0.85 * 0.975 = 0.8288
    score_low_stat = compute_ranking_score(0.60, 1.00)  # 0.60
    assert score_high_web > score_low_stat


def test_cross_source_conflict_detection():
    pipeline = VerificationPipeline()

    # Source A: Notice period is 7 days
    source_a = b"The mandatory notice period for contract termination is exactly 7 days."
    # Source B: Notice period is 10 days
    source_b = b"The mandatory notice period for contract termination is strictly 10 days."

    # Claim aligns with Source A, contradicting Source B
    draft = "The mandatory notice period for contract termination is 7 days."

    cert = pipeline.run_verification(
        draft_text=draft,
        source_files=[
            ("policy_a.txt", source_a),
            ("policy_b.txt", source_b)
        ],
        source_authorities={
            "policy_a.txt": "STATUTORY / OFFICIAL",
            "policy_b.txt": "INTERNAL / ORGANIZATIONAL"
        },
        top_k=3
    )

    assert len(cert.claims) >= 1
    claim = cert.claims[0]

    # Verify conflict was detected across sources
    assert claim.conflict_detected is True
    assert claim.primary_evidence is not None
    assert claim.primary_evidence.source == "policy_a.txt"
    assert len(claim.conflicting_evidence) >= 1
    assert any(ce.source == "policy_b.txt" for ce in claim.conflicting_evidence)

    # Escalation to REVIEW_REQUIRED
    assert cert.summary.conflicts_detected >= 1
    assert cert.overall_verdict == OverallVerdict.REVIEW_REQUIRED


def test_timing_and_cryptographic_audit_metrics():
    pipeline = VerificationPipeline()
    draft = "All reports must be submitted within 30 days."
    source = b"Mandatory submission timeline: All reports must be submitted within 30 days of fiscal close."

    cert = pipeline.run_verification(
        draft_text=draft,
        source_files=[("rules.txt", source)]
    )

    # Input hash
    assert cert.input_hash is not None
    assert len(cert.input_hash) == 64  # SHA-256 hex string

    # Timing metrics
    assert cert.execution_time_seconds >= 0.0
    assert cert.summary.total_time_seconds >= 0.0
    assert cert.summary.avg_time_per_claim_ms >= 0.0
    assert len(cert.claims) >= 1
    assert cert.claims[0].processing_time_ms >= 0.0


def test_default_authority_handling():
    pipeline = VerificationPipeline()
    draft = "Annual disclosures are required."
    source = b"Annual disclosures are required for all certified entities."

    # No source_authorities provided -> should default cleanly
    cert = pipeline.run_verification(
        draft_text=draft,
        source_files=[("handbook.txt", source)]
    )

    assert cert.sources[0].authority_level == DEFAULT_AUTHORITY_LEVEL
    assert cert.sources[0].authority_weight == DEFAULT_AUTHORITY_WEIGHT
    assert cert.claims[0].source_authority == DEFAULT_AUTHORITY_LEVEL
