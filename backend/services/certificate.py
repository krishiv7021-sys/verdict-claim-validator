import os
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from backend.schemas import (
    VerificationCertificate,
    VerificationSummary,
    OverallVerdict,
    VerdictType,
    SourceMetadata,
    ClaimVerificationResult,
    VerificationConfig
)


def determine_overall_verdict(summary: VerificationSummary) -> OverallVerdict:
    """
    Conservative verdict evaluation logic:
    If ANY claim is REFUTED or UNVERIFIED -> REVIEW_REQUIRED.
    Only if ALL claims are actively SUPPORTED -> VERIFIED.
    """
    if summary.total_claims == 0:
        return OverallVerdict.REVIEW_REQUIRED
    if summary.refuted > 0 or summary.unverified > 0:
        return OverallVerdict.REVIEW_REQUIRED
    if summary.supported == summary.total_claims:
        return OverallVerdict.VERIFIED
    return OverallVerdict.REVIEW_REQUIRED


def create_certificate(
    sources: List[SourceMetadata],
    claims: List[ClaimVerificationResult],
    config: VerificationConfig,
    certificate_id: str = None
) -> VerificationCertificate:
    """
    Assembles a complete, cryptographically anchored Verification Certificate.
    """
    if not certificate_id:
        certificate_id = f"vc_{uuid.uuid4().hex[:12]}"

    supported = sum(1 for c in claims if c.verdict == VerdictType.SUPPORTED)
    refuted = sum(1 for c in claims if c.verdict == VerdictType.REFUTED)
    unverified = sum(1 for c in claims if c.verdict == VerdictType.UNVERIFIED)

    summary = VerificationSummary(
        total_claims=len(claims),
        supported=supported,
        refuted=refuted,
        unverified=unverified
    )

    overall_verdict = determine_overall_verdict(summary)
    timestamp = datetime.now(timezone.utc).isoformat()

    return VerificationCertificate(
        certificate_version="1.0",
        certificate_id=certificate_id,
        timestamp=timestamp,
        overall_verdict=overall_verdict,
        summary=summary,
        configuration=config,
        sources=sources,
        claims=claims
    )


def save_certificate_to_disk(
    certificate: VerificationCertificate,
    storage_dir: str = "certificates"
) -> str:
    """
    Persists certificate JSON to disk for audit trails.
    """
    path = Path(storage_dir)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f"{certificate.certificate_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(certificate.model_dump_json(indent=2))
    return str(file_path)


def load_certificate_from_disk(
    certificate_id: str,
    storage_dir: str = "certificates"
) -> VerificationCertificate:
    file_path = Path(storage_dir) / f"{certificate_id}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Certificate {certificate_id} not found.")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return VerificationCertificate.model_validate(data)


def generate_human_readable_report(cert: VerificationCertificate) -> str:
    """
    Generates a clear, professional verification report in Markdown format.
    """
    badge = "🟢 VERIFIED" if cert.overall_verdict == OverallVerdict.VERIFIED else "🔴 REVIEW REQUIRED"
    
    report_lines = [
        f"# VERDICT: Evidence-Grounded Verification Report",
        f"**Certificate ID**: `{cert.certificate_id}`",
        f"**Verification Timestamp**: {cert.timestamp}",
        f"**Overall Status**: {badge}",
        f"",
        f"---",
        f"## 1. Executive Summary",
        f"| Metric | Value |",
        f"| :--- | :--- |",
        f"| Total Claims Evaluated | **{cert.summary.total_claims}** |",
        f"| Claims Directly Supported | **{cert.summary.supported}** |",
        f"| Claims Refuted / Contradicted | **{cert.summary.refuted}** |",
        f"| Unverified Claims (Ambiguous / Missing) | **{cert.summary.unverified}** |",
        f"",
        f"---",
        f"## 2. Source Documents & Cryptographic Hashes",
        f"| Document Name | Format | SHA-256 Checksum | Evidence Scope |",
        f"| :--- | :--- | :--- | :--- |"
    ]

    for src in cert.sources:
        loc_summary = f"{len(src.evidence_locations)} locations" if src.evidence_locations else f"{src.page_count} page(s)"
        report_lines.append(f"| `{src.filename}` | `{src.file_type.upper()}` | `{src.sha256[:16]}...` | {loc_summary} |")

    report_lines.extend([
        f"",
        f"---",
        f"## 3. Claim-by-Claim Verification Audit",
        f""
    ])

    for claim in cert.claims:
        if claim.verdict == VerdictType.SUPPORTED:
            v_badge = "✅ SUPPORTED"
        elif claim.verdict == VerdictType.REFUTED:
            v_badge = "❌ REFUTED"
        else:
            v_badge = "⚠️ UNVERIFIED"

        report_lines.append(f"### Claim [{claim.claim_id}]: {v_badge}")
        report_lines.append(f"**AI Claim**: \"{claim.text}\"")
        report_lines.append(f"**Verdict Rationale**: {claim.reason}")
        report_lines.append(f"**Retrieval Score**: `{claim.retrieval_score:.2f}` | **Entailment Confidence**: `{claim.confidence:.2f}`")

        if claim.evidence:
            top_ev = claim.evidence[0]
            loc_label = top_ev.location or (f"Page {top_ev.page}" if top_ev.page else "N/A")
            report_lines.append(f"**Primary Evidence Grounding**:")
            report_lines.append(f"> \"{top_ev.text}\"")
            report_lines.append(f"> — *Source: `{top_ev.source}` ({loc_label})*")
        else:
            report_lines.append(f"*No matching evidence passage found in source materials.*")

        report_lines.append(f"")

    report_lines.extend([
        f"---",
        f"## 4. Audit Metadata & Reproducibility",
        f"- **Embedding Model**: `{cert.configuration.embedding_model}`",
        f"- **Entailment Provider**: `{cert.configuration.entailment_provider}` ({cert.configuration.entailment_model})",
        f"- **Top-K Retrieval**: `{cert.configuration.top_k}`",
        f"- **Specification**: VERDICT Certificate v{cert.certificate_version}",
        f"",
        f"*Generated by VERDICT: Claim Validato (Team Cygnix)*"
    ])

    return "\n".join(report_lines)
