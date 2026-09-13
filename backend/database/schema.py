import logging
from typing import Optional
from backend.database.connection import get_db_connection

logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = 1

SCHEMA_SQL = """
-- Schema Migrations Table
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

-- Verifications Table
CREATE TABLE IF NOT EXISTS verifications (
    id TEXT PRIMARY KEY,
    certificate_id TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL,
    overall_status TEXT NOT NULL,
    claim_count INTEGER NOT NULL,
    supported_count INTEGER NOT NULL,
    refuted_count INTEGER NOT NULL,
    unverified_count INTEGER NOT NULL,
    conflicts_detected INTEGER NOT NULL DEFAULT 0,
    total_sources INTEGER NOT NULL DEFAULT 0,
    execution_time_seconds REAL NOT NULL DEFAULT 0.0,
    avg_time_per_claim_ms REAL NOT NULL DEFAULT 0.0,
    input_hash TEXT,
    draft_snippet TEXT
);

CREATE INDEX IF NOT EXISTS idx_verifications_created_at ON verifications(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_verifications_cert_id ON verifications(certificate_id);

-- Claims Table
CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY,
    verification_id TEXT NOT NULL,
    claim_index INTEGER NOT NULL,
    claim_text TEXT NOT NULL,
    original_sentence TEXT,
    verdict TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.0,
    reason TEXT,
    similarity_score REAL NOT NULL DEFAULT 0.0,
    ranking_score REAL NOT NULL DEFAULT 0.0,
    source_authority TEXT,
    conflict_detected INTEGER NOT NULL DEFAULT 0,
    processing_time_ms REAL NOT NULL DEFAULT 0.0,
    FOREIGN KEY (verification_id) REFERENCES verifications(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_claims_verification_id ON claims(verification_id);
CREATE INDEX IF NOT EXISTS idx_claims_verdict ON claims(verdict);

-- Evidence Table
CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL,
    source_filename TEXT NOT NULL,
    source_id TEXT,
    location TEXT,
    location_type TEXT,
    page INTEGER,
    snippet_text TEXT NOT NULL,
    similarity REAL NOT NULL DEFAULT 0.0,
    ranking_score REAL NOT NULL DEFAULT 0.0,
    authority_level TEXT,
    authority_weight REAL NOT NULL DEFAULT 1.0,
    is_primary INTEGER NOT NULL DEFAULT 0,
    is_conflict INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (claim_id) REFERENCES claims(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_evidence_claim_id ON evidence(claim_id);

-- Certificates Table
CREATE TABLE IF NOT EXISTS certificates (
    certificate_id TEXT PRIMARY KEY,
    verification_id TEXT NOT NULL,
    certificate_version TEXT NOT NULL DEFAULT '1.1',
    created_at TEXT NOT NULL,
    overall_verdict TEXT NOT NULL,
    input_hash TEXT,
    raw_json TEXT NOT NULL,
    sha256_hash TEXT NOT NULL,
    FOREIGN KEY (verification_id) REFERENCES verifications(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_certificates_verification_id ON certificates(verification_id);
"""


def init_db(db_path: Optional[str] = None) -> None:
    """
    Initializes database tables and indices if they do not already exist.
    Preserves all existing data. Idempotent and thread-safe.
    """
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.executescript(SCHEMA_SQL)
        
        # Check and record schema version
        cursor.execute("SELECT MAX(version) FROM schema_migrations;")
        row = cursor.fetchone()
        latest_ver = row[0] if row and row[0] is not None else 0
        if latest_ver < CURRENT_SCHEMA_VERSION:
            from datetime import datetime, timezone
            now_iso = datetime.now(timezone.utc).isoformat()
            cursor.execute(
                "INSERT OR REPLACE INTO schema_migrations (version, applied_at) VALUES (?, ?);",
                (CURRENT_SCHEMA_VERSION, now_iso)
            )
        conn.commit()
        logger.info(f"Database initialized successfully (schema v{CURRENT_SCHEMA_VERSION}).")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to initialize database schema: {e}")
        raise
    finally:
        conn.close()
