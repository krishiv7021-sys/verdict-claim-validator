import os
import logging
from datetime import datetime, timezone
from typing import List, Tuple

from backend.schemas import (
    SourceMetadata,
    SourceChunk,
    ClaimVerificationResult,
    VerificationConfig,
    VerificationCertificate,
    VerdictType
)
from backend.services.document_parser import parse_document
from backend.services.claim_decomposer import decompose_draft
from backend.services.retriever import retrieve_evidence_for_claims
from backend.services.entailment import get_entailment_engine
from backend.services.certificate import create_certificate, save_certificate_to_disk

logger = logging.getLogger(__name__)


class VerificationPipeline:
    """
    End-to-End Evidence-Grounded Claim Verification Engine.
    Coordinates document parsing, claim decomposition, embedding retrieval,
    strict entailment verification, and certificate generation.
    """
    def __init__(self):
        self.entailment_engine = get_entailment_engine()

    def run_verification(
        self,
        draft_text: str,
        source_files: List[Tuple[str, bytes]],
        top_k: int = 3,
        similarity_threshold: float = 0.25
    ) -> VerificationCertificate:
        """
        Executes end-to-end verification.
        Gracefully handles empty inputs, unparseable files, or edge cases without crashing.
        """
        sources_meta: List[SourceMetadata] = []
        all_chunks: List[SourceChunk] = []

        # 1. Parse each source document
        for filename, file_bytes in source_files:
            try:
                meta, chunks = parse_document(file_bytes, filename)
                sources_meta.append(meta)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error parsing source {filename}: {e}")

        # 2. Decompose AI draft into atomic claims
        draft_clean = (draft_text or "").strip()
        if not draft_clean:
            claims = []
        else:
            claims = decompose_draft(draft_clean)

        # 3. Retrieve evidence chunks for all claims
        retrieval_map = retrieve_evidence_for_claims(
            claims,
            all_chunks,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

        # 4. Verify each claim with the entailment engine
        verified_claims: List[ClaimVerificationResult] = []
        now_ts = datetime.now(timezone.utc).isoformat()

        for claim in claims:
            evidence_spans = retrieval_map.get(claim.claim_id, [])
            top_score = evidence_spans[0].similarity if evidence_spans else 0.0

            if not all_chunks:
                # No source material provided or extracted
                verdict = VerdictType.UNVERIFIED
                confidence = 0.0
                reason = "No extractable text or evidence was available from uploaded source documents."
            else:
                verdict, confidence, reason = self.entailment_engine.verify_claim(
                    claim.text,
                    evidence_spans
                )

            verified_claims.append(
                ClaimVerificationResult(
                    claim_id=claim.claim_id,
                    text=claim.text,
                    original_sentence=claim.original_sentence,
                    verdict=verdict,
                    confidence=confidence,
                    reason=reason,
                    retrieval_score=top_score,
                    evidence=evidence_spans,
                    verified_at=now_ts
                )
            )

        # 5. Build configuration metadata
        provider_name = os.getenv("LLM_PROVIDER", "heuristic_fallback")
        config = VerificationConfig(
            embedding_model="BAAI/bge-small-en-v1.5",
            entailment_provider=provider_name,
            entailment_model=os.getenv(f"{provider_name.upper()}_MODEL", "default"),
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            certificate_version="1.0"
        )

        # 6. Generate certificate
        certificate = create_certificate(
            sources=sources_meta,
            claims=verified_claims,
            config=config
        )

        # 7. Persist certificate to disk
        try:
            save_certificate_to_disk(certificate)
        except Exception as e:
            logger.warning(f"Could not save certificate to disk: {e}")

        return certificate
