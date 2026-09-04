from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class VerdictType(str, Enum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    UNVERIFIED = "UNVERIFIED"


class OverallVerdict(str, Enum):
    VERIFIED = "VERIFIED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class SourceChunk(BaseModel):
    source_id: str
    filename: str
    page: int = 1
    chunk_id: str
    text: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    file_type: str = "txt"
    location_type: str = "page"
    location: str = "Page 1"


class SourceMetadata(BaseModel):
    source_id: str
    filename: str
    sha256: str
    file_type: str
    page_count: int = 1
    char_count: int = 0
    size: int = 0
    evidence_locations: List[str] = Field(default_factory=list)


class AtomicClaim(BaseModel):
    claim_id: str
    text: str
    original_sentence: Optional[str] = None


class EvidenceSpan(BaseModel):
    source: str
    source_id: Optional[str] = None
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    text: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    similarity: float = 0.0
    file_type: Optional[str] = None
    location_type: Optional[str] = "page"
    location: Optional[str] = None


class ClaimVerificationResult(BaseModel):
    claim_id: str
    text: str
    original_sentence: Optional[str] = None
    verdict: VerdictType
    confidence: float = 0.0
    reason: str = ""
    retrieval_score: float = 0.0
    evidence: List[EvidenceSpan] = Field(default_factory=list)
    verified_at: str = ""


class VerificationSummary(BaseModel):
    total_claims: int = 0
    supported: int = 0
    refuted: int = 0
    unverified: int = 0


class VerificationConfig(BaseModel):
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    entailment_provider: str = "heuristic_fallback"
    entailment_model: str = "default"
    top_k: int = 3
    similarity_threshold: float = 0.35
    certificate_version: str = "1.0"


class VerificationCertificate(BaseModel):
    certificate_version: str = "1.0"
    certificate_id: str
    timestamp: str
    overall_verdict: OverallVerdict
    summary: VerificationSummary
    configuration: VerificationConfig = Field(default_factory=VerificationConfig)
    sources: List[SourceMetadata] = Field(default_factory=list)
    claims: List[ClaimVerificationResult] = Field(default_factory=list)


class VerifyResponse(BaseModel):
    status: str = "success"
    certificate: VerificationCertificate
    draft_text: str
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    embedding_model: str
    entailment_provider: str
    timestamp: str
