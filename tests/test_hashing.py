import hashlib
from backend.services.hashing import compute_sha256, compute_file_sha256


def test_compute_sha256_string():
    content = "Verifiable AI Output"
    expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert compute_sha256(content) == expected


def test_compute_sha256_bytes():
    content_b = b"Test document content"
    expected = hashlib.sha256(content_b).hexdigest()
    assert compute_sha256(content_b) == expected


def test_sha256_tamper_detection():
    doc1 = b"All filings are due in 30 days."
    doc2 = b"All filings are due in 90 days."
    assert compute_sha256(doc1) != compute_sha256(doc2)


def test_compute_file_sha256(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("Hello verification audit", encoding="utf-8")
    expected = hashlib.sha256(b"Hello verification audit").hexdigest()
    assert compute_file_sha256(str(f)) == expected
