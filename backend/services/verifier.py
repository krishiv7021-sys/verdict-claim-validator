import os
import time
import logging
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict

from backend.schemas import (
    SourceMetadata,
    SourceChunk,
    ClaimVerificationResult,
    VerificationConfig,
    VerificationCertificate,
    VerdictType,
    normalize_authority_level
)
from backend.services.hashing import compute_sha256
from backend.services.document_parser import parse_document
from backend.services.claim_decomposer import decompose_draft
from backend.services.retriever import retrieve_evidence_for_claims
from backend.services.entailment import get_entailment_engine
from backend.services.certificate import create_certificate, save_certificate_to_disk

logger = logging.getLogger(__name__)


class VerificationPipeline:
    """
    End-to-End Evidence-Grounded Claim Verification Engine.
    Coordinates document parsing with authority levels, claim decomposition,
    authority-weighted semantic retrieval, cross-source conflict detection,
    performance tracking, and reproducible cryptographic certification.
    """
    def __init__(self):
        self.entailment_engine = get_entailment_engine()

    def run_verification(
        self,
        draft_text: str,
        source_files: List[Tuple[str, bytes]],
        top_k: int = 3,
        similarity_threshold: float = 0.25,
        source_authorities: Optional[Dict[str, str]] = None,
        enable_authority_weighting: bool = True,
        enable_conflict_detection: bool = True
    ) -> VerificationCertificate:
        """
        Executes end-to-end verification.
        Gracefully handles empty inputs, unparseable files, edge cases, and authority assignments.
        """
        start_time = time.perf_counter()
        sources_meta: List[SourceMetadata] = []
        all_chunks: List[SourceChunk] = []

        # 1. Parse each source document with authority attribution
        for filename, file_bytes in source_files:
            try:
                meta, chunks = parse_document(file_bytes, filename)
                
                # Determine authority level and weight
                raw_auth = source_authorities.get(filename) if source_authorities else None
                auth_level, auth_weight = normalize_authority_level(raw_auth)
                
                meta.authority_level = auth_level
                meta.authority_weight = auth_weight

                for chunk in chunks:
                    chunk.authority_level = auth_level
                    chunk.authority_weight = auth_weight

                sources_meta.append(meta)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error parsing source {filename}: {e}")

        # 2. Decompose AI draft into atomic claims
        draft_clean = (draft_text or "").strip()
        input_hash = compute_sha256(draft_clean.encode("utf-8")) if draft_clean else None

        if not draft_clean:
            claims = []
        else:
            claims = decompose_draft(draft_clean)

        # 3. Retrieve evidence chunks for all claims with authority weighting
        retrieval_map = retrieve_evidence_for_claims(
            claims,
            all_chunks,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            enable_authority_weighting=enable_authority_weighting
        )

        # 4. Verify each claim with entailment engine and detect conflicts
        verified_claims: List[ClaimVerificationResult] = []
        now_ts = datetime.now(timezone.utc).isoformat()

        for claim in claims:
            claim_start = time.perf_counter()
            evidence_spans = retrieval_map.get(claim.claim_id, [])

            if not all_chunks:
                verdict = VerdictType.UNVERIFIED
                confidence = 0.0
                reason = "No extractable text or evidence was available from uploaded source documents."
                eval_res = {
                    "verdict": verdict,
                    "confidence": confidence,
                    "reason": reason,
                    "primary_evidence": None,
                    "supporting_evidence": [],
                    "conflicting_evidence": [],
                    "conflict_detected": False
                }
            else:
                if enable_conflict_detection and hasattr(self.entailment_engine, "evaluate_candidates"):
                    eval_res = self.entailment_engine.evaluate_candidates(claim.text, evidence_spans)
                else:
                    v, conf, r = self.entailment_engine.verify_claim(claim.text, evidence_spans)
                    eval_res = {
                        "verdict": v,
                        "confidence": conf,
                        "reason": r,
                        "primary_evidence": evidence_spans[0] if evidence_spans else None,
                        "supporting_evidence": [evidence_spans[0]] if evidence_spans and v == VerdictType.SUPPORTED else [],
                        "conflicting_evidence": [evidence_spans[0]] if evidence_spans and v == VerdictType.REFUTED else [],
                        "conflict_detected": False
                    }

            claim_proc_time_ms = round((time.perf_counter() - claim_start) * 1000.0, 2)
            primary_ev = eval_res.get("primary_evidence") or (evidence_spans[0] if evidence_spans else None)
            top_similarity = primary_ev.similarity if primary_ev else 0.0
            top_ranking = primary_ev.ranking_score if primary_ev else 0.0
            auth_str = primary_ev.authority_level if primary_ev else None

            verified_claims.append(
                ClaimVerificationResult(
                    claim_id=claim.claim_id,
                    text=claim.text,
                    claim_text=claim.text,
                    original_sentence=claim.original_sentence,
                    verdict=eval_res["verdict"],
                    confidence=eval_res["confidence"],
                    reason=eval_res["reason"],
                    explanation=eval_res["reason"],
                    retrieval_score=top_similarity,
                    similarity_score=top_similarity,
                    ranking_score=top_ranking,
                    evidence=evidence_spans,
                    primary_evidence=primary_ev,
                    supporting_evidence=eval_res.get("supporting_evidence", []),
                    conflicting_evidence=eval_res.get("conflicting_evidence", []),
                    source_authority=auth_str,
                    conflict_detected=eval_res.get("conflict_detected", False),
                    verified_at=now_ts,
                    processing_time_ms=claim_proc_time_ms
                )
            )

        # 5. Measure performance metrics
        total_time_seconds = round(time.perf_counter() - start_time, 4)
        avg_time_per_claim_ms = round((total_time_seconds / len(claims) * 1000.0) if claims else 0.0, 2)
        conflicts_count = sum(1 for c in verified_claims if c.conflict_detected)

        # 6. Build configuration metadata
        provider_name = os.getenv("LLM_PROVIDER", "heuristic_fallback")
        config = VerificationConfig(
            embedding_model="BAAI/bge-small-en-v1.5",
            entailment_provider=provider_name,
            entailment_model=os.getenv(f"{provider_name.upper()}_MODEL", "default"),
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            certificate_version="1.1",
            authority_weighting_enabled=enable_authority_weighting,
            conflict_detection_enabled=enable_conflict_detection
        )

        # 7. Generate certificate
        certificate = create_certificate(
            sources=sources_meta,
            claims=verified_claims,
            config=config,
            input_hash=input_hash,
            total_time_seconds=total_time_seconds,
            avg_time_per_claim_ms=avg_time_per_claim_ms,
            conflicts_detected=conflicts_count
        )

        # 8. Persist certificate to disk
        try:
            save_certificate_to_disk(certificate)
        except Exception as e:
            logger.warning(f"Could not save certificate to disk: {e}")

        return certificate
