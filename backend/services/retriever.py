from typing import List, Dict, Optional
import numpy as np
from backend.schemas import AtomicClaim, SourceChunk, EvidenceSpan
from backend.services.embeddings import get_embedding_service


def compute_ranking_score(similarity: float, authority_weight: float, enable_weighting: bool = True) -> float:
    """
    Computes combined ranking score balancing semantic relevance and source authority.
    Formula: similarity * (0.75 + 0.25 * authority_weight)
    Ensures semantic relevance is dominant and authority cannot promote an irrelevant passage.
    """
    if not enable_weighting:
        return round(similarity, 4)
    # Clamp inputs
    sim = max(0.0, min(1.0, similarity))
    w = max(0.5, min(1.0, authority_weight))
    # Semantic relevance is 75% baseline weight, authority accounts for up to 25% boost
    return round(sim * (0.75 + 0.25 * w), 4)


def retrieve_evidence_for_claims(
    claims: List[AtomicClaim],
    source_chunks: List[SourceChunk],
    top_k: int = 3,
    similarity_threshold: float = 0.25,
    enable_authority_weighting: bool = True
) -> Dict[str, List[EvidenceSpan]]:
    """
    Retrieves top-k relevant candidate evidence chunks for each claim using vector embeddings
    and authority-weighted ranking.
    Returns a dictionary mapping claim_id -> list of EvidenceSpan objects.
    """
    retrieval_map: Dict[str, List[EvidenceSpan]] = {c.claim_id: [] for c in claims}

    if not claims or not source_chunks:
        return retrieval_map

    embedder = get_embedding_service()

    # Extract texts
    chunk_texts = [c.text for c in source_chunks]
    claim_texts = [c.text for c in claims]

    # Batch embed
    chunk_vectors = embedder.embed_texts(chunk_texts)
    claim_vectors = embedder.embed_texts(claim_texts)

    if chunk_vectors.shape[0] == 0 or claim_vectors.shape[0] == 0:
        return retrieval_map

    # Ensure 2D arrays
    if len(chunk_vectors.shape) == 1:
        chunk_vectors = chunk_vectors.reshape(1, -1)
    if len(claim_vectors.shape) == 1:
        claim_vectors = claim_vectors.reshape(1, -1)

    # Compute cosine similarity matrix (claims x chunks)
    similarity_matrix = np.dot(claim_vectors, chunk_vectors.T)

    for i, claim in enumerate(claims):
        scores = similarity_matrix[i]
        
        # Calculate ranking score for each chunk
        scored_candidates = []
        for idx in range(len(source_chunks)):
            sim = float(scores[idx])
            chunk = source_chunks[idx]
            r_score = compute_ranking_score(sim, chunk.authority_weight, enable_authority_weighting)
            scored_candidates.append((r_score, sim, idx))

        # Sort primarily by ranking_score descending, secondarily by raw similarity
        scored_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

        candidates: List[EvidenceSpan] = []
        for r_score, sim, idx in scored_candidates:
            if sim >= similarity_threshold or len(candidates) == 0:
                chunk = source_chunks[idx]
                candidates.append(
                    EvidenceSpan(
                        source=chunk.filename,
                        source_id=chunk.source_id,
                        page=chunk.page,
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        start_char=chunk.start_char,
                        end_char=chunk.end_char,
                        similarity=round(sim, 4),
                        semantic_score=round(sim, 4),
                        ranking_score=r_score,
                        file_type=chunk.file_type,
                        location_type=chunk.location_type,
                        location=chunk.location,
                        authority_level=chunk.authority_level,
                        authority_weight=chunk.authority_weight
                    )
                )
            if len(candidates) >= top_k:
                break

        retrieval_map[claim.claim_id] = candidates

    return retrieval_map
