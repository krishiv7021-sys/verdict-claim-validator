import io
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "VERDICT: Claim Validato"
    assert data["team"] == "Cygnix"
    assert data["status"] == "online"


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "embedding_model" in data
    assert "entailment_provider" in data


def test_api_demo():
    response = client.get("/demo")
    assert response.status_code == 200
    data = response.json()
    assert "draft_text" in data
    assert "source_filename" in data
    assert "source_content" in data


def test_api_verify_empty_draft():
    response = client.post("/verify", data={"draft_text": ""})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "error" in data


def test_api_verify_with_sources():
    draft = "Annual compliance filings are due within 30 days. Penalty is $50,000."
    source_content = b"Mandatory Deadline: All annual filings must be completed within 30 days. Late penalty is $15,000 per violation."

    files = [
        ("source_files", ("compliance.txt", io.BytesIO(source_content), "text/plain"))
    ]
    data = {"draft_text": draft}

    response = client.post("/verify", data=data, files=files)
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "success"
    cert = result["certificate"]
    assert cert["certificate_id"].startswith("vc_")
    assert len(cert["sources"]) == 1
    assert cert["sources"][0]["filename"] == "compliance.txt"
    assert len(cert["claims"]) >= 2
    assert cert["overall_verdict"] == "REVIEW_REQUIRED"


def test_api_certificate_and_report_retrieval():
    draft = "Filings are due within 30 days."
    source_content = b"Filings are due within 30 days."
    files = [
        ("source_files", ("doc.txt", io.BytesIO(source_content), "text/plain"))
    ]
    verify_resp = client.post("/verify", data={"draft_text": draft}, files=files)
    cert_id = verify_resp.json()["certificate"]["certificate_id"]

    # Retrieve certificate
    cert_resp = client.get(f"/certificate/{cert_id}")
    assert cert_resp.status_code == 200
    assert cert_resp.json()["certificate_id"] == cert_id

    # Retrieve human-readable report
    report_resp = client.get(f"/certificate/{cert_id}/report")
    assert report_resp.status_code == 200
    assert "VERDICT: Evidence-Grounded Verification Report" in report_resp.text
