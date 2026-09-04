from typing import List, Dict
import numpy as np
from backend.schemas import AtomicClaim, SourceChunk, EvidenceSpan
from backend.services.embeddings import get_embedding_service


def retrieve_evidence_for_claims(
    claims: List[AtomicClaim],
    source_chunks: List[SourceChunk],
    top_k: int = 3,
    similarity_threshold: float = 0.25
) -> Dict[str, List[EvidenceSpan]]:
    """
    Retrieves top-k relevant candidate evidence chunks for each claim using vector embeddings.
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
    # Both vectors are already normalized by the embedder
    similarity_matrix = np.dot(claim_vectors, chunk_vectors.T)

    for i, claim in enumerate(claims):
        scores = similarity_matrix[i]
        
        # Rank indices descending
        ranked_indices = np.argsort(scores)[::-1]
        
        candidates: List[EvidenceSpan] = []
        for idx in ranked_indices[:top_k]:
            sim = float(scores[idx])
            # Only include candidate if it meets the minimum similarity floor
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
                        file_type=chunk.file_type,
                        location_type=chunk.location_type,
                        location=chunk.location
                    )
                )

        retrieval_map[claim.claim_id] = candidates

    return retrieval_map
