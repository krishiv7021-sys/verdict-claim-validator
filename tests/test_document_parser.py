import io
from backend.services.document_parser import (
    parse_txt_document,
    parse_pdf_document,
    parse_document,
    split_into_sentences,
    create_chunks_from_text
)


def test_split_into_sentences():
    text = "First statement is true. Second statement is also verified! What about the third?"
    spans = split_into_sentences(text)
    assert len(spans) == 3
    assert "First statement is true." in spans[0][0]
    assert spans[0][1] >= 0
    assert spans[0][2] > spans[0][1]


def test_parse_txt_document():
    raw_text = """Header Section
This is paragraph one with important compliance rules.

This is paragraph two with penalties and exemptions.
"""
    file_bytes = raw_text.encode("utf-8")
    meta, chunks = parse_txt_document(file_bytes, "policy.txt")

    assert meta.filename == "policy.txt"
    assert meta.file_type == "txt"
    assert len(meta.sha256) == 64
    assert len(chunks) >= 2
    assert chunks[0].page == 1
    assert chunks[0].filename == "policy.txt"


def test_empty_document_handling():
    meta, chunks = parse_txt_document(b"", "empty.txt")
    assert meta.filename == "empty.txt"
    assert len(chunks) == 0


def test_synthetic_pdf_parsing():
    import pypdf
    # Create an in-memory PDF using pypdf
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    pdf_buffer = io.BytesIO()
    writer.write(pdf_buffer)
    pdf_bytes = pdf_buffer.getvalue()

    meta, chunks = parse_pdf_document(pdf_bytes, "sample.pdf")
    assert meta.filename == "sample.pdf"
    assert meta.file_type == "pdf"
    assert meta.page_count >= 1


def test_real_pdf_file_parsing():
    from pathlib import Path
    pdf_path = Path(__file__).parent.parent / "data" / "sample_policy.pdf"
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        meta, chunks = parse_pdf_document(pdf_bytes, "sample_policy.pdf")
        assert meta.filename == "sample_policy.pdf"
        assert meta.file_type == "pdf"
        assert len(chunks) >= 1
        combined_text = " ".join(c.text for c in chunks)
        assert "30 days" in combined_text
        assert "15,000" in combined_text
