import re
import uuid
import sqlite3
import logging
from typing import Optional, List, Dict, Any

from backend.schemas import VerificationCertificate
from backend.services.hashing import compute_sha256
from backend.database.connection import get_db_context

logger = logging.getLogger(__name__)

SAFE_ID_REGEX = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')


def save_verification(
    cert: VerificationCertificate,
    draft_text: Optional[str] = None,
    db_path: Optional[str] = None
) -> str:
    """
    Atomically persists a complete verification run including claims,
    evidence metadata, and cryptographic certificate inside a single transaction.
    Rolls back completely if any failure occurs.
    """
    if not cert or not cert.certificate_id:
        raise ValueError("Cannot persist verification: invalid certificate object.")

    cert_id = cert.certificate_id
    if not SAFE_ID_REGEX.match(cert_id):
        raise ValueError(f"Invalid certificate_id format: {cert_id}")

    verification_id = f"verif_{cert_id.replace('vc_', '')}"
    created_at = cert.timestamp
    overall_status = cert.overall_verdict.value
    claim_count = cert.summary.total_claims
    supported_count = cert.summary.supported
    refuted_count = cert.summary.refuted
    unverified_count = cert.summary.unverified
    conflicts_detected = cert.summary.conflicts_detected
    total_sources = cert.summary.total_sources
    exec_time = cert.summary.total_time_seconds
    avg_claim_ms = cert.summary.avg_time_per_claim_ms
    input_hash = cert.input_hash

    # Safe draft snippet for UI history preview (capped, no sensitive control characters)
    draft_snippet = None
    if draft_text:
        clean_draft = " ".join(draft_text.split()).strip()
        draft_snippet = clean_draft[:250] + ("..." if len(clean_draft) > 250 else "")

    raw_json = cert.model_dump_json(indent=2)
    sha256_hash = compute_sha256(raw_json.encode("utf-8"))

    # Auto-initialize tables if missing
    try:
        from backend.database.schema import init_db
        init_db(db_path)
    except Exception as e:
        logger.debug(f"Schema init check: {e}")

    with get_db_context(db_path) as conn:
        cursor = conn.cursor()

        # 1. Insert Verification Record
        cursor.execute(
            """
            INSERT INTO verifications (
                id, certificate_id, created_at, overall_status,
                claim_count, supported_count, refuted_count, unverified_count,
                conflicts_detected, total_sources, execution_time_seconds,
                avg_time_per_claim_ms, input_hash, draft_snippet
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                verification_id, cert_id, created_at, overall_status,
                claim_count, supported_count, refuted_count, unverified_count,
                conflicts_detected, total_sources, exec_time,
                avg_claim_ms, input_hash, draft_snippet
            )
        )

        # 2. Insert Claims and Evidence
        for idx, claim in enumerate(cert.claims):
            db_claim_id = f"{verification_id}_clm_{idx+1}"
            cursor.execute(
                """
                INSERT INTO claims (
                    id, verification_id, claim_index, claim_text,
                    original_sentence, verdict, confidence, reason,
                    similarity_score, ranking_score, source_authority,
                    conflict_detected, processing_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    db_claim_id,
                    verification_id,
                    idx + 1,
                    claim.text,
                    claim.original_sentence,
                    claim.verdict.value,
                    float(claim.confidence),
                    claim.reason,
                    float(claim.similarity_score),
                    float(claim.ranking_score),
                    claim.source_authority,
                    1 if claim.conflict_detected else 0,
                    float(claim.processing_time_ms)
                )
            )

            # Insert Evidence Spans (deduplicated by chunk/text/source)
            inserted_evidence_keys = set()
            ev_counter = 0

            # Gather all evidence collections
            all_ev_candidates = []
            if claim.primary_evidence:
                all_ev_candidates.append((claim.primary_evidence, True, False))
            for ev in claim.supporting_evidence:
                all_ev_candidates.append((ev, False, False))
            for ev in claim.evidence:
                all_ev_candidates.append((ev, False, False))
            for ev in claim.conflicting_evidence:
                all_ev_candidates.append((ev, False, True))

            for ev, is_prim, is_conf in all_ev_candidates:
                ev_key = (ev.source, ev.location or ev.page, ev.text[:60])
                if ev_key in inserted_evidence_keys:
                    continue
                inserted_evidence_keys.add(ev_key)

                ev_counter += 1
                ev_id = f"{db_claim_id}_ev_{ev_counter}"
                snippet_text = ev.text.strip()[:1000]

                cursor.execute(
                    """
                    INSERT INTO evidence (
                        id, claim_id, source_filename, source_id,
                        location, location_type, page, snippet_text,
                        similarity, ranking_score, authority_level,
                        authority_weight, is_primary, is_conflict
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        ev_id,
                        db_claim_id,
                        ev.source,
                        ev.source_id,
                        ev.location or (f"Page {ev.page}" if ev.page else "N/A"),
                        ev.location_type or "page",
                        ev.page if ev.page else 1,
                        snippet_text,
                        float(ev.similarity),
                        float(ev.ranking_score),
                        ev.authority_level,
                        float(ev.authority_weight),
                        1 if is_prim else 0,
                        1 if is_conf else 0
                    )
                )

        # 3. Insert Certificate
        cursor.execute(
            """
            INSERT INTO certificates (
                certificate_id, verification_id, certificate_version,
                created_at, overall_verdict, input_hash, raw_json, sha256_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                cert_id,
                verification_id,
                cert.certificate_version,
                created_at,
                overall_status,
                input_hash,
                raw_json,
                sha256_hash
            )
        )

    logger.info(f"Persisted verification {verification_id} (Cert: {cert_id}) to SQLite.")
    return verification_id


def get_verification(verification_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieves a complete verification record including claims, evidence, and certificate metadata.
    Accepts either a verification_id ('verif_...') or certificate_id ('vc_...').
    """
    if not verification_id or not SAFE_ID_REGEX.match(verification_id):
        return None

    try:
        with get_db_context(db_path) as conn:
            cursor = conn.cursor()

            # Query verifications table
            cursor.execute(
                """
                SELECT * FROM verifications WHERE id = ? OR certificate_id = ?;
                """,
                (verification_id, verification_id)
            )
            row = cursor.fetchone()
            if not row:
                return None

            v_dict = dict(row)
            actual_vid = v_dict["id"]

            # Fetch claims
            cursor.execute(
                """
                SELECT * FROM claims WHERE verification_id = ? ORDER BY claim_index ASC;
                """,
                (actual_vid,)
            )
            claims = []
            for clm_row in cursor.fetchall():
                clm_dict = dict(clm_row)
                clm_dict["claim_id"] = f"C{clm_dict['claim_index']:03d}"
                # Fetch evidence for each claim
                cursor.execute(
                    """
                    SELECT * FROM evidence WHERE claim_id = ? ORDER BY is_primary DESC, ranking_score DESC;
                    """,
                    (clm_dict["id"],)
                )
                clm_dict["evidence"] = [dict(ev) for ev in cursor.fetchall()]
                claims.append(clm_dict)

            v_dict["claims"] = claims

            # Fetch certificate summary
            cursor.execute(
                """
                SELECT certificate_id, certificate_version, created_at, overall_verdict, sha256_hash
                FROM certificates WHERE verification_id = ?;
                """,
                (actual_vid,)
            )
            cert_row = cursor.fetchone()
            v_dict["certificate"] = dict(cert_row) if cert_row else None

            return v_dict
    except sqlite3.OperationalError as e:
        logger.debug(f"Database query failed (table missing or uninitialized): {e}")
        return None


def get_certificate_from_db(
    certificate_id: str,
    db_path: Optional[str] = None
) -> Optional[VerificationCertificate]:
    """
    Retrieves and deserializes a VerificationCertificate from the SQLite database.
    Returns None if not found, table missing, or if the certificate_id is invalid.
    """
    if not certificate_id or not SAFE_ID_REGEX.match(certificate_id):
        return None

    try:
        with get_db_context(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT raw_json FROM certificates WHERE certificate_id = ?;",
                (certificate_id,)
            )
            row = cursor.fetchone()
            if not row or not row["raw_json"]:
                return None

            try:
                return VerificationCertificate.model_validate_json(row["raw_json"])
            except Exception as e:
                logger.error(f"Error deserializing certificate {certificate_id} from DB: {e}")
                return None
    except sqlite3.OperationalError as e:
        logger.debug(f"Database table uninitialized in get_certificate_from_db: {e}")
        return None


def list_verifications(
    limit: int = 50,
    offset: int = 0,
    db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Returns historical verification records sorted by recency.
    Enforces safe bounds on limit and offset.
    """
    safe_limit = max(1, min(limit, 200))
    safe_offset = max(0, offset)

    try:
        with get_db_context(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    id,
                    certificate_id,
                    created_at,
                    overall_status,
                    claim_count,
                    supported_count,
                    refuted_count,
                    unverified_count,
                    conflicts_detected,
                    total_sources,
                    execution_time_seconds,
                    draft_snippet
                FROM verifications
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?;
                """,
                (safe_limit, safe_offset)
            )
            return [dict(row) for row in cursor.fetchall()]
    except sqlite3.OperationalError as e:
        logger.debug(f"Database table uninitialized in list_verifications: {e}")
        return []


def get_verification_count(db_path: Optional[str] = None) -> int:
    """
    Returns total number of stored verification runs.
    """
    try:
        with get_db_context(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM verifications;")
            row = cursor.fetchone()
            return row[0] if row else 0
    except sqlite3.OperationalError as e:
        logger.debug(f"Database table uninitialized in get_verification_count: {e}")
        return 0


def delete_verification(verification_id: str, db_path: Optional[str] = None) -> bool:
    """
    Deletes a verification record by ID. Foreign key cascades delete associated claims,
    evidence, and certificates.
    """
    if not verification_id or not SAFE_ID_REGEX.match(verification_id):
        return False

    with get_db_context(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM verifications WHERE id = ? OR certificate_id = ?;",
            (verification_id, verification_id)
        )
        return cursor.rowcount > 0
