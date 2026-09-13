import io
import os
import sqlite3
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from backend.main import app
from backend.database.connection import get_db_connection, get_db_context, check_db_health
from backend.database.schema import init_db
from backend.database.repository import (
    save_verification,
    get_verification,
    get_certificate_from_db,
    list_verifications,
    get_verification_count,
    delete_verification
)
from backend.services.certificate import (
    create_certificate,
    save_certificate_to_disk,
    load_certificate
)
from backend.schemas import (
    VerificationConfig,
    SourceMetadata,
    ClaimVerificationResult,
    EvidenceSpan,
    VerdictType,
    OverallVerdict
)
from backend.services.verifier import VerificationPipeline


@pytest.fixture
def temp_db(tmp_path):
    """Provides a dedicated temporary SQLite database for each test."""
    db_path = str(tmp_path / "test_verdict.db")
    init_db(db_path)
    return db_path


def make_dummy_certificate(cert_id="vc_test123456", overall=OverallVerdict.VERIFIED):
    """Helper to construct a realistic VerificationCertificate for database tests."""
    sources = [
        SourceMetadata(
            source_id="src_1",
            filename="policy.txt",
            sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            file_type="txt",
            page_count=1,
            char_count=150,
            size=150,
            authority_level="STATUTORY / OFFICIAL",
            authority_weight=1.00
        )
    ]
    evidence = [
        EvidenceSpan(
            source="policy.txt",
            source_id="src_1",
            page=1,
            location="Page 1",
            text="Filing deadline is exactly 30 days.",
            similarity=0.92,
            ranking_score=0.92,
            authority_level="STATUTORY / OFFICIAL",
            authority_weight=1.00
        )
    ]
    claims = [
        ClaimVerificationResult(
            claim_id="claim_1",
            text="Filing deadline is 30 days.",
            original_sentence="Filing deadline is 30 days.",
            verdict=VerdictType.SUPPORTED,
            confidence=0.95,
            reason="Direct statutory match.",
            similarity_score=0.92,
            ranking_score=0.92,
            evidence=evidence,
            primary_evidence=evidence[0],
            source_authority="STATUTORY / OFFICIAL",
            conflict_detected=False,
            processing_time_ms=12.5
        )
    ]
    config = VerificationConfig(
        embedding_model="BAAI/bge-small-en-v1.5",
        entailment_provider="heuristic_fallback",
        top_k=3,
        similarity_threshold=0.25
    )
    return create_certificate(
        sources=sources,
        claims=claims,
        config=config,
        certificate_id=cert_id,
        input_hash="d04b98f48e8f8bcc15c6ae5ac050801cd6dcfd428fb5f9e65c4e16e7807340fa",
        total_time_seconds=0.15,
        avg_time_per_claim_ms=150.0
    )


# =====================================================================
# 1. Database Initialization & Table Creation
# =====================================================================
def test_db_initialization_and_tables(temp_db):
    conn = get_db_connection(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {row[0] for row in cursor.fetchall()}
    conn.close()

    assert "verifications" in tables
    assert "claims" in tables
    assert "evidence" in tables
    assert "certificates" in tables
    assert "schema_migrations" in tables


def test_idempotent_init_db(temp_db):
    # Calling init_db multiple times should not raise errors or duplicate tables
    init_db(temp_db)
    init_db(temp_db)
    assert check_db_health(temp_db) is True


# =====================================================================
# 2. Saving and Retrieving Verifications
# =====================================================================
def test_save_and_retrieve_verification(temp_db):
    cert = make_dummy_certificate("vc_unit_001")
    draft = "Filing deadline is 30 days."

    verif_id = save_verification(cert, draft_text=draft, db_path=temp_db)
    assert verif_id == "verif_unit_001"

    # Retrieve by verification_id
    record = get_verification("verif_unit_001", db_path=temp_db)
    assert record is not None
    assert record["id"] == "verif_unit_001"
    assert record["certificate_id"] == "vc_unit_001"
    assert record["overall_status"] == "VERIFIED"
    assert record["claim_count"] == 1
    assert record["supported_count"] == 1
    assert record["draft_snippet"] == "Filing deadline is 30 days."
    assert len(record["claims"]) == 1

    # Retrieve by certificate_id alias
    record_by_cert = get_verification("vc_unit_001", db_path=temp_db)
    assert record_by_cert is not None
    assert record_by_cert["id"] == "verif_unit_001"


# =====================================================================
# 3. Claims & Evidence Metadata Storage
# =====================================================================
def test_claims_and_evidence_metadata_persistence(temp_db):
    cert = make_dummy_certificate("vc_clm_ev_001")
    save_verification(cert, db_path=temp_db)

    record = get_verification("vc_clm_ev_001", db_path=temp_db)
    claim = record["claims"][0]
    assert claim["claim_text"] == "Filing deadline is 30 days."
    assert claim["verdict"] == "SUPPORTED"
    assert claim["confidence"] == 0.95
    assert claim["source_authority"] == "STATUTORY / OFFICIAL"
    assert len(claim["evidence"]) == 1

    ev = claim["evidence"][0]
    assert ev["source_filename"] == "policy.txt"
    assert ev["location"] == "Page 1"
    assert "Filing deadline" in ev["snippet_text"]
    assert ev["authority_level"] == "STATUTORY / OFFICIAL"
    assert ev["is_primary"] == 1


# =====================================================================
# 4. Certificate Metadata & Deserialization
# =====================================================================
def test_certificate_storage_and_deserialization(temp_db):
    cert = make_dummy_certificate("vc_cert_store_01")
    save_verification(cert, db_path=temp_db)

    cert_from_db = get_certificate_from_db("vc_cert_store_01", db_path=temp_db)
    assert cert_from_db is not None
    assert cert_from_db.certificate_id == "vc_cert_store_01"
    assert cert_from_db.overall_verdict == OverallVerdict.VERIFIED
    assert len(cert_from_db.claims) == 1
    assert cert_from_db.claims[0].text == "Filing deadline is 30 days."


# =====================================================================
# 5. History Listing & Pagination
# =====================================================================
def test_verification_history_pagination(temp_db):
    for i in range(5):
        cert = make_dummy_certificate(f"vc_hist_{i:03d}")
        save_verification(cert, draft_text=f"Draft number {i}", db_path=temp_db)

    assert get_verification_count(db_path=temp_db) == 5

    page1 = list_verifications(limit=3, offset=0, db_path=temp_db)
    assert len(page1) == 3

    page2 = list_verifications(limit=3, offset=3, db_path=temp_db)
    assert len(page2) == 2

    # Check safe clamping on excessive limits
    clamped = list_verifications(limit=999, offset=0, db_path=temp_db)
    assert len(clamped) == 5


# =====================================================================
# 6. Missing Record Handling
# =====================================================================
def test_missing_record_handling(temp_db):
    assert get_verification("vc_non_existent", db_path=temp_db) is None
    assert get_certificate_from_db("vc_non_existent", db_path=temp_db) is None
    assert delete_verification("vc_non_existent", db_path=temp_db) is False


# =====================================================================
# 7. Duplicate ID Handling
# =====================================================================
def test_duplicate_id_handling(temp_db):
    cert = make_dummy_certificate("vc_dup_001")
    save_verification(cert, db_path=temp_db)

    # Re-inserting the same certificate must raise sqlite3.IntegrityError
    with pytest.raises(sqlite3.IntegrityError):
        save_verification(cert, db_path=temp_db)


# =====================================================================
# 8. Atomic Transaction Rollback on Failure
# =====================================================================
def test_transaction_rollback_on_failure(temp_db):
    cert = make_dummy_certificate("vc_rollback_test")

    # Corrupt the certificate in a way that fails during claim/evidence processing
    # but passes initial checks: set an invalid attribute that raises an error
    class BadClaim:
        text = "test"
        original_sentence = None
        verdict = VerdictType.SUPPORTED
        confidence = "not_a_float"  # Will trigger float conversion error in repository

    cert.claims.append(BadClaim())

    with pytest.raises(Exception):
        save_verification(cert, db_path=temp_db)

    # Verify zero records were written (entire transaction rolled back)
    record = get_verification("vc_rollback_test", db_path=temp_db)
    assert record is None
    assert get_verification_count(db_path=temp_db) == 0


# =====================================================================
# 9. Database Restart / Persistence
# =====================================================================
def test_database_restart_persistence(tmp_path):
    db_file = str(tmp_path / "persistent_restart.db")
    init_db(db_file)

    cert = make_dummy_certificate("vc_restart_test")
    save_verification(cert, draft_text="Draft survives restart.", db_path=db_file)

    # Simulate restart by checking health and querying on fresh connection
    assert check_db_health(db_file) is True
    reopened_cert = get_certificate_from_db("vc_restart_test", db_path=db_file)
    assert reopened_cert is not None
    assert reopened_cert.certificate_id == "vc_restart_test"


# =====================================================================
# 10. SQL Injection Protections
# =====================================================================
def test_sql_injection_defense(temp_db):
    injection_cert_id = "vc_test'; DROP TABLE verifications; --"
    # Should be rejected by safe ID regex
    with pytest.raises(ValueError):
        cert = make_dummy_certificate(injection_cert_id)
        save_verification(cert, db_path=temp_db)

    # Injection in draft text should be safely escaped as literal parameterized content
    safe_cert = make_dummy_certificate("vc_safe_inject_test")
    sql_draft = "Legitimate draft'; DROP TABLE claims; SELECT * FROM 'users"
    save_verification(safe_cert, draft_text=sql_draft, db_path=temp_db)

    # Ensure tables still exist and query returned correctly
    record = get_verification("vc_safe_inject_test", db_path=temp_db)
    assert record is not None
    assert record["draft_snippet"] == sql_draft
    assert check_db_health(temp_db) is True


# =====================================================================
# 11. Malformed and Path Traversal IDs
# =====================================================================
def test_invalid_and_path_traversal_ids(temp_db):
    malformed_ids = [
        "../../etc/passwd",
        "vc_test/../traversal",
        "vc_test\x00nullbyte",
        "",
        "   ",
        "vc_" + ("a" * 100)  # Too long
    ]
    for bad_id in malformed_ids:
        assert get_verification(bad_id, db_path=temp_db) is None
        assert get_certificate_from_db(bad_id, db_path=temp_db) is None


# =====================================================================
# 12. Delete Cascades
# =====================================================================
def test_delete_cascade(temp_db):
    cert = make_dummy_certificate("vc_cascade_test")
    save_verification(cert, db_path=temp_db)

    assert get_verification("vc_cascade_test", db_path=temp_db) is not None
    deleted = delete_verification("vc_cascade_test", db_path=temp_db)
    assert deleted is True

    # Check cascading deletes
    conn = get_db_connection(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM claims WHERE verification_id = 'verif_cascade_test';")
    assert cursor.fetchone()[0] == 0
    cursor.execute("SELECT COUNT(*) FROM certificates WHERE certificate_id = 'vc_cascade_test';")
    assert cursor.fetchone()[0] == 0
    conn.close()


# =====================================================================
# 13. Certificate Fallback to Disk
# =====================================================================
def test_certificate_fallback_to_disk(tmp_path):
    cert_dir = tmp_path / "legacy_certs"
    cert = make_dummy_certificate("vc_legacy_disk_001")
    save_certificate_to_disk(cert, storage_dir=str(cert_dir))

    # Certificate is NOT in database, but is on disk
    loaded = load_certificate("vc_legacy_disk_001", storage_dir=str(cert_dir))
    assert loaded is not None
    assert loaded.certificate_id == "vc_legacy_disk_001"


# =====================================================================
# 14. API History & Details Endpoints
# =====================================================================
def test_api_history_endpoints(temp_db, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", temp_db)
    client = TestClient(app)

    # Empty history initially
    resp = client.get("/verifications")
    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "items" in data

    # Perform a verification via API
    draft = "Filings are mandatory within 30 days."
    source_content = b"Mandatory Deadline: All filings must be completed within 30 days."
    files = [("source_files", ("directive.txt", io.BytesIO(source_content), "text/plain"))]

    v_resp = client.post("/verify", data={"draft_text": draft}, files=files)
    assert v_resp.status_code == 200
    cert_id = v_resp.json()["certificate"]["certificate_id"]

    # History should now include this verification
    resp2 = client.get("/verifications?limit=10")
    assert resp2.status_code == 200
    history = resp2.json()
    assert history["total"] >= 1
    matching = [item for item in history["items"] if item["certificate_id"] == cert_id]
    assert len(matching) == 1
    assert matching[0]["claim_count"] >= 1

    # Verification detail endpoint
    verif_id = matching[0]["id"]
    detail_resp = client.get(f"/verifications/{verif_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["id"] == verif_id
    assert len(detail["claims"]) >= 1

    # Health check reflects database connectivity
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["database"] == "connected"


# =====================================================================
# 15. Frontend History Inspection Interaction
# =====================================================================
def test_frontend_history_inspection_interaction(temp_db, monkeypatch):
    """
    Verifies that inspecting a verification record in the frontend correctly:
    1. Loads the certificate into session_state.
    2. Renders the active inspection panel in tab_history.
    3. Allows closing the inspection panel cleanly.
    """
    monkeypatch.setenv("DATABASE_PATH", temp_db)
    
    # Pre-populate temp_db with a verification record
    cert = make_dummy_certificate(cert_id="vc_inspect_test_99", overall=OverallVerdict.VERIFIED)
    save_verification(cert, draft_text="Filing deadline is 30 days.", db_path=temp_db)
    
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("frontend/app.py", default_timeout=15)
    at.run()
    assert not at.exception
    
    # Locate the inspect button for our record
    inspect_btns = [b for b in at.button if b.key and b.key.startswith("btn_load_")]
    assert len(inspect_btns) >= 1
    
    # Click inspect
    target_btn = inspect_btns[0]
    target_btn.click().run()
    assert not at.exception
    
    # Verify session state was populated
    assert at.session_state["history_inspected_cert"] is not None
    assert at.session_state["history_inspected_cert"]["certificate_id"] == "vc_inspect_test_99"
    assert at.session_state["verification_result"] is not None
    
    # Verify close inspection button is now rendered and active
    close_btns = [b for b in at.button if b.key == "btn_close_inspection"]
    assert len(close_btns) == 1
    
    # Verify claim inspection buttons rendered in history tab
    hist_claim_btns = [b for b in at.button if b.key and b.key.startswith("hist_btn_")]
    assert len(hist_claim_btns) >= 1
    
    # Click close inspection
    close_btns[0].click().run()
    assert not at.exception
    assert at.session_state["history_inspected_cert"] is None
    assert len([b for b in at.button if b.key == "btn_close_inspection"]) == 0

