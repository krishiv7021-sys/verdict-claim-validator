from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field, model_validator


class VerdictType(str, Enum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    UNVERIFIED = "UNVERIFIED"


class OverallVerdict(str, Enum):
    VERIFIED = "VERIFIED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class SourceAuthorityLevel(str, Enum):
    STATUTORY = "STATUTORY / OFFICIAL"
    POLICY = "POLICY / REGULATION"
    INTERNAL = "INTERNAL / ORGANIZATIONAL"
    REFERENCE = "REFERENCE / WEB"


AUTHORITY_WEIGHTS: Dict[str, float] = {
    SourceAuthorityLevel.STATUTORY.value: 1.00,
    SourceAuthorityLevel.POLICY.value: 0.97,
    SourceAuthorityLevel.INTERNAL.value: 0.94,
    SourceAuthorityLevel.REFERENCE.value: 0.90,
}

DEFAULT_AUTHORITY_LEVEL = SourceAuthorityLevel.INTERNAL.value
DEFAULT_AUTHORITY_WEIGHT = AUTHORITY_WEIGHTS[DEFAULT_AUTHORITY_LEVEL]


def normalize_authority_level(raw: Optional[str]) -> Tuple[str, float]:
    """
    Normalizes raw authority string input into canonical authority level and weight.
    Extensible: handles abbreviations, partial matches, and provides clean fallback.
    """
    if not raw:
        return DEFAULT_AUTHORITY_LEVEL, DEFAULT_AUTHORITY_WEIGHT
    s = raw.strip().upper()
    for level, weight in AUTHORITY_WEIGHTS.items():
        if s == level.upper():
            return level, weight
    if "STATUTORY" in s or "OFFICIAL" in s:
        return SourceAuthorityLevel.STATUTORY.value, 1.00
    if "POLICY" in s or "REGULATION" in s:
        return SourceAuthorityLevel.POLICY.value, 0.97
    if "INTERNAL" in s or "ORGANIZATIONAL" in s:
        return SourceAuthorityLevel.INTERNAL.value, 0.94
    if "REFERENCE" in s or "WEB" in s:
        return SourceAuthorityLevel.REFERENCE.value, 0.90
    return DEFAULT_AUTHORITY_LEVEL, DEFAULT_AUTHORITY_WEIGHT


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
    authority_level: str = DEFAULT_AUTHORITY_LEVEL
    authority_weight: float = DEFAULT_AUTHORITY_WEIGHT


class SourceMetadata(BaseModel):
    source_id: str
    filename: str
    sha256: str
    file_type: str
    page_count: int = 1
    char_count: int = 0
    size: int = 0
    evidence_locations: List[str] = Field(default_factory=list)
    authority_level: str = DEFAULT_AUTHORITY_LEVEL
    authority_weight: float = DEFAULT_AUTHORITY_WEIGHT


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
    semantic_score: float = 0.0
    ranking_score: float = 0.0
    file_type: Optional[str] = None
    location_type: Optional[str] = "page"
    location: Optional[str] = None
    authority_level: str = DEFAULT_AUTHORITY_LEVEL
    authority_weight: float = DEFAULT_AUTHORITY_WEIGHT

    @model_validator(mode="after")
    def populate_scores(self):
        if self.semantic_score == 0.0 and self.similarity != 0.0:
            self.semantic_score = self.similarity
        if self.ranking_score == 0.0 and self.similarity != 0.0:
            self.ranking_score = self.similarity
        return self


class ClaimVerificationResult(BaseModel):
    claim_id: str
    text: str
    claim_text: Optional[str] = None
    original_sentence: Optional[str] = None
    verdict: VerdictType
    confidence: float = 0.0
    reason: str = ""
    explanation: Optional[str] = None
    retrieval_score: float = 0.0
    similarity_score: float = 0.0
    ranking_score: float = 0.0
    evidence: List[EvidenceSpan] = Field(default_factory=list)
    primary_evidence: Optional[EvidenceSpan] = None
    supporting_evidence: List[EvidenceSpan] = Field(default_factory=list)
    conflicting_evidence: List[EvidenceSpan] = Field(default_factory=list)
    source_authority: Optional[str] = None
    conflict_detected: bool = False
    verified_at: str = ""
    processing_time_ms: float = 0.0

    @model_validator(mode="after")
    def populate_aliases(self):
        if not self.claim_text:
            self.claim_text = self.text
        if not self.explanation:
            self.explanation = self.reason
        if self.similarity_score == 0.0 and self.retrieval_score != 0.0:
            self.similarity_score = self.retrieval_score
        if self.ranking_score == 0.0 and self.retrieval_score != 0.0:
            self.ranking_score = self.retrieval_score
        if not self.primary_evidence and self.evidence:
            self.primary_evidence = self.evidence[0]
            if not self.source_authority:
                self.source_authority = self.evidence[0].authority_level
        return self


class VerificationSummary(BaseModel):
    total_claims: int = 0
    supported: int = 0
    refuted: int = 0
    unverified: int = 0
    conflicts_detected: int = 0
    total_sources: int = 0
    total_time_seconds: float = 0.0
    avg_time_per_claim_ms: float = 0.0


class VerificationConfig(BaseModel):
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    entailment_provider: str = "heuristic_fallback"
    entailment_model: str = "default"
    top_k: int = 3
    similarity_threshold: float = 0.25
    certificate_version: str = "1.1"
    authority_weighting_enabled: bool = True
    conflict_detection_enabled: bool = True


class VerificationCertificate(BaseModel):
    certificate_version: str = "1.1"
    certificate_id: str
    timestamp: str
    overall_verdict: OverallVerdict
    summary: VerificationSummary
    configuration: VerificationConfig = Field(default_factory=VerificationConfig)
    sources: List[SourceMetadata] = Field(default_factory=list)
    claims: List[ClaimVerificationResult] = Field(default_factory=list)
    input_hash: Optional[str] = None
    execution_time_seconds: float = 0.0


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
