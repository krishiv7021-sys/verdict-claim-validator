import io
import re
import csv
import json
from typing import List, Tuple, Optional, Dict, Any

import logging

from backend.schemas import SourceMetadata, SourceChunk
from backend.services.hashing import compute_sha256
from backend.utils.security import (
    validate_file_type_and_content,
    sanitize_filename,
    log_security_event
)

logger = logging.getLogger("verdict_parser")


# Graceful optional imports
try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None

try:
    import pptx
except ImportError:
    pptx = None

try:
    import openpyxl
except ImportError:
    openpyxl = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


def split_into_sentences(text: str) -> List[Tuple[str, int, int]]:
    """
    Splits text into sentences while tracking start and end character offsets.
    Returns list of (sentence_text, start_char, end_char).
    """
    sentence_spans = []
    pattern = re.compile(r'([A-Z0-9\$\"\'\(][^\.\?!]*[\.\?!]+(?=[\s\n]|$)|\S[^\.\?!]*$)', re.DOTALL)
    for match in pattern.finditer(text):
        s_text = match.group(0).strip()
        if s_text and len(s_text) > 10:
            start = match.start()
            end = match.end()
            sentence_spans.append((s_text, start, end))

    if not sentence_spans and text.strip():
        sentence_spans.append((text.strip(), 0, len(text)))

    return sentence_spans


def create_chunks_from_text(
    text: str,
    source_id: str,
    filename: str,
    page_num: int = 1,
    target_chunk_size: int = 300,
    file_type: str = "txt",
    location_type: str = "page",
    default_location: Optional[str] = None
) -> List[SourceChunk]:
    """
    Segments page or document text into coherent chunks (1-3 sentences or paragraphs).
    Preserves exact character offsets, location_type, and location string.
    """
    chunks: List[SourceChunk] = []
    if not text or not text.strip():
        return chunks

    lines = text.splitlines()
    paragraphs = []
    curr_block = []
    for line in lines:
        line_str = line.strip()
        if not line_str:
            if curr_block:
                paragraphs.append(" ".join(curr_block))
                curr_block = []
            continue
        if re.match(r'^(?:Section\s+\d+|[0-9]+[\.\)]|Rule\s+\d+|Article\s+\d+)', line_str, re.IGNORECASE):
            if curr_block:
                paragraphs.append(" ".join(curr_block))
                curr_block = []
        curr_block.append(line_str)
    if curr_block:
        paragraphs.append(" ".join(curr_block))

    current_char_offset = 0
    chunk_idx = 1
    for para_idx, para in enumerate(paragraphs):
        para_clean = para.strip()
        if not para_clean:
            continue

        start_in_doc = text.find(para, current_char_offset)
        if start_in_doc == -1:
            start_in_doc = current_char_offset

        # Determine location display
        if default_location:
            loc_str = default_location
        elif location_type == "page":
            loc_str = f"Page {page_num}"
        elif location_type == "paragraph":
            loc_str = f"Paragraph {para_idx + 1}"
        else:
            loc_str = f"Section {para_idx + 1}"

        if len(para_clean) <= target_chunk_size:
            chunk = SourceChunk(
                source_id=source_id,
                filename=filename,
                page=page_num,
                chunk_id=f"{source_id}-P{page_num}-C{chunk_idx:03d}",
                text=para_clean,
                start_char=start_in_doc,
                end_char=start_in_doc + len(para_clean),
                file_type=file_type,
                location_type=location_type,
                location=loc_str
            )
            chunks.append(chunk)
            chunk_idx += 1
        else:
            sents = split_into_sentences(para_clean)
            cur_sub_text = []
            sub_start = start_in_doc

            for s_text, s_start, s_end in sents:
                cur_sub_text.append(s_text)
                combined = " ".join(cur_sub_text)
                if len(combined) >= target_chunk_size:
                    chunks.append(SourceChunk(
                        source_id=source_id,
                        filename=filename,
                        page=page_num,
                        chunk_id=f"{source_id}-P{page_num}-C{chunk_idx:03d}",
                        text=combined,
                        start_char=sub_start,
                        end_char=sub_start + len(combined),
                        file_type=file_type,
                        location_type=location_type,
                        location=loc_str
                    ))
                    chunk_idx += 1
                    cur_sub_text = []
                    sub_start = start_in_doc + s_end

            if cur_sub_text:
                combined = " ".join(cur_sub_text)
                chunks.append(SourceChunk(
                    source_id=source_id,
                    filename=filename,
                    page=page_num,
                    chunk_id=f"{source_id}-P{page_num}-C{chunk_idx:03d}",
                    text=combined,
                    start_char=sub_start,
                    end_char=sub_start + len(combined),
                    file_type=file_type,
                    location_type=location_type,
                    location=loc_str
                ))
                chunk_idx += 1

        current_char_offset = start_in_doc + len(para) + 1

    return chunks


# =====================================================================
# 1. PDF Parser
# =====================================================================
def parse_pdf(file_bytes: bytes, filename: str) -> Tuple[SourceMetadata, List[SourceChunk]]:
    """Extracts text per page from PDF bytes using pypdf."""
    sha256_hash = compute_sha256(file_bytes)
    source_id = f"SRC_{sha256_hash[:8]}"
    chunks: List[SourceChunk] = []
    total_chars = 0
    page_count = 0

    try:
        if pypdf is None:
            text = file_bytes.decode("utf-8", errors="ignore")
            chunks = create_chunks_from_text(
                text, source_id, filename, page_num=1,
                file_type="pdf", location_type="page", default_location="Page 1"
            )
            page_count = 1
            total_chars = len(text)
        else:
            pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            page_count = len(pdf_reader.pages)

            for idx, page in enumerate(pdf_reader.pages):
                page_num = idx + 1
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""

                page_text = page_text.strip()
                total_chars += len(page_text)
                if page_text:
                    page_chunks = create_chunks_from_text(
                        page_text, source_id, filename, page_num=page_num,
                        file_type="pdf", location_type="page", default_location=f"Page {page_num}"
                    )
                    chunks.extend(page_chunks)
    except Exception:
        pass

    locations = sorted(list(set(c.location for c in chunks)))
    metadata = SourceMetadata(
        source_id=source_id,
        filename=filename,
        sha256=sha256_hash,
        file_type="pdf",
        page_count=max(page_count, 1),
        char_count=total_chars,
        size=len(file_bytes),
        evidence_locations=locations
    )
    return metadata, chunks

# Backward-compatibility alias
parse_pdf_document = parse_pdf


# =====================================================================
# 2. TXT Parser
# =====================================================================
def parse_txt(file_bytes: bytes, filename: str) -> Tuple[SourceMetadata, List[SourceChunk]]:
    """Extracts text from plain text files."""
    sha256_hash = compute_sha256(file_bytes)
    source_id = f"SRC_{sha256_hash[:8]}"

    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1", errors="replace")

    chunks = create_chunks_from_text(
        text, source_id, filename, page_num=1,
        file_type="txt", location_type="paragraph"
    )

    locations = sorted(list(set(c.location for c in chunks)))
    metadata = SourceMetadata(
        source_id=source_id,
        filename=filename,
        sha256=sha256_hash,
        file_type="txt",
        page_count=1,
        char_count=len(text),
        size=len(file_bytes),
        evidence_locations=locations
    )
    return metadata, chunks

# Backward-compatibility alias
parse_txt_document = parse_txt


# =====================================================================
# 3. DOCX Parser
# =====================================================================
def parse_docx(file_bytes: bytes, filename: str) -> Tuple[SourceMetadata, List[SourceChunk]]:
    """Extracts paragraphs and tables from Word DOCX documents."""
    sha256_hash = compute_sha256(file_bytes)
    source_id = f"SRC_{sha256_hash[:8]}"
    chunks: List[SourceChunk] = []
    total_chars = 0

    if docx is None:
        return parse_txt(file_bytes, filename)

    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        chunk_idx = 1

        # Extract paragraphs
        for p_idx, para in enumerate(doc.paragraphs):
            p_text = para.text.strip()
            if not p_text:
                continue

            total_chars += len(p_text)
            loc_str = f"Paragraph {p_idx + 1}"
            chunk = SourceChunk(
                source_id=source_id,
                filename=filename,
                page=1,
                chunk_id=f"{source_id}-PARA{p_idx+1:03d}-C{chunk_idx:03d}",
                text=p_text,
                start_char=None,
                end_char=None,
                file_type="docx",
                location_type="paragraph",
                location=loc_str
            )
            chunks.append(chunk)
            chunk_idx += 1

        # Extract tables
        for t_idx, table in enumerate(doc.tables):
            for r_idx, row in enumerate(table.rows):
                row_texts = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if not row_texts:
                    continue
                row_str = " | ".join(row_texts)
                total_chars += len(row_str)
                loc_str = f"Table {t_idx + 1}, Row {r_idx + 1}"
                chunk = SourceChunk(
                    source_id=source_id,
                    filename=filename,
                    page=1,
                    chunk_id=f"{source_id}-TBL{t_idx+1}-R{r_idx+1}",
                    text=row_str,
                    start_char=None,
                    end_char=None,
                    file_type="docx",
                    location_type="table",
                    location=loc_str
                )
                chunks.append(chunk)
                chunk_idx += 1

    except Exception:
        pass

    locations = sorted(list(set(c.location for c in chunks)))
    metadata = SourceMetadata(
        source_id=source_id,
        filename=filename,
        sha256=sha256_hash,
        file_type="docx",
        page_count=1,
        char_count=total_chars,
        size=len(file_bytes),
        evidence_locations=locations
    )
    return metadata, chunks


# =====================================================================
# 4. PPTX Parser
# =====================================================================
def parse_pptx(file_bytes: bytes, filename: str) -> Tuple[SourceMetadata, List[SourceChunk]]:
    """Extracts slides and text shapes from PowerPoint PPTX presentations."""
    sha256_hash = compute_sha256(file_bytes)
    source_id = f"SRC_{sha256_hash[:8]}"
    chunks: List[SourceChunk] = []
    total_chars = 0
    slide_count = 0

    if pptx is None:
        return parse_txt(file_bytes, filename)

    try:
        prs = pptx.Presentation(io.BytesIO(file_bytes))
        slide_count = len(prs.slides)
        chunk_idx = 1

        for s_idx, slide in enumerate(prs.slides):
            slide_num = s_idx + 1
            slide_texts = []

            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        text = para.text.strip()
                        if text:
                            slide_texts.append(text)

            if slide_texts:
                combined_text = "\n".join(slide_texts)
                total_chars += len(combined_text)
                loc_str = f"Slide {slide_num}"
                
                # Split large slides or keep as single chunk per slide
                chunk = SourceChunk(
                    source_id=source_id,
                    filename=filename,
                    page=slide_num,
                    chunk_id=f"{source_id}-SLIDE{slide_num:03d}-C{chunk_idx:03d}",
                    text=combined_text,
                    start_char=None,
                    end_char=None,
                    file_type="pptx",
                    location_type="slide",
                    location=loc_str
                )
                chunks.append(chunk)
                chunk_idx += 1

    except Exception:
        pass

    locations = sorted(list(set(c.location for c in chunks)))
    metadata = SourceMetadata(
        source_id=source_id,
        filename=filename,
        sha256=sha256_hash,
        file_type="pptx",
        page_count=max(slide_count, 1),
        char_count=total_chars,
        size=len(file_bytes),
        evidence_locations=locations
    )
    return metadata, chunks


# =====================================================================
# 5. XLSX Parser
# =====================================================================
def parse_xlsx(file_bytes: bytes, filename: str) -> Tuple[SourceMetadata, List[SourceChunk]]:
    """Extracts rows and cells from Excel XLSX workbooks."""
    sha256_hash = compute_sha256(file_bytes)
    source_id = f"SRC_{sha256_hash[:8]}"
    chunks: List[SourceChunk] = []
    total_chars = 0
    sheet_count = 0

    if openpyxl is None:
        return parse_txt(file_bytes, filename)

    try:
        wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
        sheet_count = len(wb.sheetnames)
        chunk_idx = 1

        for s_idx, sheet_name in enumerate(wb.sheetnames):
            ws = wb[sheet_name]
            headers = []

            for r_idx, row in enumerate(ws.iter_rows(values_only=True)):
                row_vals = [str(val).strip() for val in row if val is not None and str(val).strip()]
                if not row_vals:
                    continue

                row_num = r_idx + 1
                if row_num == 1 and not headers:
                    headers = row_vals
                    header_str = f"Headers: {', '.join(headers)}"
                    loc_str = f"{sheet_name}!Row 1"
                    chunks.append(SourceChunk(
                        source_id=source_id,
                        filename=filename,
                        page=s_idx + 1,
                        chunk_id=f"{source_id}-S{s_idx+1}-R{row_num}",
                        text=f"Sheet '{sheet_name}' {header_str}",
                        start_char=None,
                        end_char=None,
                        file_type="xlsx",
                        location_type="cell",
                        location=loc_str
                    ))
                    chunk_idx += 1
                    continue

                # Format row text with headers if available
                if headers and len(headers) == len(row):
                    formatted_pairs = [f"{h}: {v}" for h, v in zip(headers, row) if v is not None]
                    row_str = f"[{sheet_name}] " + " | ".join(formatted_pairs)
                else:
                    row_str = f"[{sheet_name}] " + " | ".join(row_vals)

                total_chars += len(row_str)
                loc_str = f"{sheet_name}!Row {row_num}"
                chunks.append(SourceChunk(
                    source_id=source_id,
                    filename=filename,
                    page=s_idx + 1,
                    chunk_id=f"{source_id}-S{s_idx+1}-R{row_num}",
                    text=row_str,
                    start_char=None,
                    end_char=None,
                    file_type="xlsx",
                    location_type="cell",
                    location=loc_str
                ))
                chunk_idx += 1

    except Exception:
        pass

    locations = sorted(list(set(c.location for c in chunks)))
    metadata = SourceMetadata(
        source_id=source_id,
        filename=filename,
        sha256=sha256_hash,
        file_type="xlsx",
        page_count=max(sheet_count, 1),
        char_count=total_chars,
        size=len(file_bytes),
        evidence_locations=locations
    )
    return metadata, chunks


# =====================================================================
# 6. CSV Parser
# =====================================================================
def parse_csv(file_bytes: bytes, filename: str) -> Tuple[SourceMetadata, List[SourceChunk]]:
    """Extracts structured rows from CSV files."""
    sha256_hash = compute_sha256(file_bytes)
    source_id = f"SRC_{sha256_hash[:8]}"
    chunks: List[SourceChunk] = []
    total_chars = 0

    try:
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1", errors="replace")

        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        headers = []
        chunk_idx = 1

        for r_idx, row in enumerate(rows):
            clean_row = [c.strip() for c in row if c.strip()]
            if not clean_row:
                continue

            row_num = r_idx + 1
            if row_num == 1:
                headers = clean_row
                loc_str = f"Row {row_num} (Header)"
                row_str = "Columns: " + ", ".join(headers)
            else:
                if headers and len(headers) == len(row):
                    formatted_pairs = [f"{h}: {val}" for h, val in zip(headers, row) if val.strip()]
                    row_str = " | ".join(formatted_pairs)
                else:
                    row_str = " | ".join(clean_row)
                loc_str = f"Row {row_num}"

            total_chars += len(row_str)
            chunks.append(SourceChunk(
                source_id=source_id,
                filename=filename,
                page=1,
                chunk_id=f"{source_id}-ROW{row_num:04d}",
                text=row_str,
                start_char=None,
                end_char=None,
                file_type="csv",
                location_type="row",
                location=loc_str
            ))
            chunk_idx += 1

    except Exception:
        pass

    locations = sorted(list(set(c.location for c in chunks)))
    metadata = SourceMetadata(
        source_id=source_id,
        filename=filename,
        sha256=sha256_hash,
        file_type="csv",
        page_count=1,
        char_count=total_chars,
        size=len(file_bytes),
        evidence_locations=locations
    )
    return metadata, chunks


# =====================================================================
# 7. Markdown Parser (.md)
# =====================================================================
def parse_markdown(file_bytes: bytes, filename: str) -> Tuple[SourceMetadata, List[SourceChunk]]:
    """Extracts sections and headers from Markdown documents."""
    sha256_hash = compute_sha256(file_bytes)
    source_id = f"SRC_{sha256_hash[:8]}"
    chunks: List[SourceChunk] = []
    total_chars = 0

    try:
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1", errors="replace")

        lines = text.splitlines()
        current_heading = "Document Root"
        current_lines = []
        chunk_idx = 1

        def flush_section(heading: str, lines_list: List[str]):
            nonlocal chunk_idx, total_chars
            content = "\n".join(lines_list).strip()
            if content:
                total_chars += len(content)
                loc_str = f"Section '{heading}'"
                chunks.append(SourceChunk(
                    source_id=source_id,
                    filename=filename,
                    page=1,
                    chunk_id=f"{source_id}-MD{chunk_idx:03d}",
                    text=content,
                    start_char=None,
                    end_char=None,
                    file_type="md",
                    location_type="heading",
                    location=loc_str
                ))
                chunk_idx += 1

        for line in lines:
            line_str = line.strip()
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line_str)
            if heading_match:
                if current_lines:
                    flush_section(current_heading, current_lines)
                    current_lines = []
                current_heading = heading_match.group(2).strip()
                current_lines.append(line_str)
            else:
                if line_str:
                    current_lines.append(line_str)

        if current_lines:
            flush_section(current_heading, current_lines)

    except Exception:
        pass

    locations = sorted(list(set(c.location for c in chunks)))
    metadata = SourceMetadata(
        source_id=source_id,
        filename=filename,
        sha256=sha256_hash,
        file_type="md",
        page_count=1,
        char_count=total_chars,
        size=len(file_bytes),
        evidence_locations=locations
    )
    return metadata, chunks


# =====================================================================
# 8. JSON Parser
# =====================================================================
def parse_json(file_bytes: bytes, filename: str) -> Tuple[SourceMetadata, List[SourceChunk]]:
    """Extracts structured paths and key-values from JSON documents."""
    sha256_hash = compute_sha256(file_bytes)
    source_id = f"SRC_{sha256_hash[:8]}"
    chunks: List[SourceChunk] = []
    total_chars = 0

    try:
        data = json.loads(file_bytes.decode("utf-8", errors="replace"))
        chunk_idx = 1
        max_depth = 20
        max_chunks = 1000

        def traverse(node: Any, current_path: str, depth: int = 1):
            nonlocal chunk_idx, total_chars
            if depth > max_depth or chunk_idx > max_chunks:
                return

            if isinstance(node, dict):
                # If small dictionary, emit as unified chunk
                if len(node) <= 6 and all(not isinstance(v, (dict, list)) for v in node.values()):
                    summary = " | ".join(f"{k}: {v}" for k, v in node.items())
                    text_str = f"Path {current_path}: {summary}"
                    total_chars += len(text_str)
                    chunks.append(SourceChunk(
                        source_id=source_id,
                        filename=filename,
                        page=1,
                        chunk_id=f"{source_id}-JSON{chunk_idx:03d}",
                        text=text_str,
                        start_char=None,
                        end_char=None,
                        file_type="json",
                        location_type="json_path",
                        location=current_path
                    ))
                    chunk_idx += 1
                else:
                    for k, v in node.items():
                        new_path = f"{current_path}.{k}" if current_path != "$" else f"$.{k}"
                        traverse(v, new_path, depth + 1)
            elif isinstance(node, list):
                for i, item in enumerate(node):
                    new_path = f"{current_path}[{i}]"
                    traverse(item, new_path, depth + 1)
            else:
                text_str = f"{current_path}: {node}"
                total_chars += len(text_str)
                chunks.append(SourceChunk(
                    source_id=source_id,
                    filename=filename,
                    page=1,
                    chunk_id=f"{source_id}-JSON{chunk_idx:03d}",
                    text=text_str,
                    start_char=None,
                    end_char=None,
                    file_type="json",
                    location_type="json_path",
                    location=current_path
                ))
                chunk_idx += 1

        traverse(data, "$", 1)

    except Exception as e:
        logger.warning(f"Error parsing JSON document '{filename}': {e}")


    locations = sorted(list(set(c.location for c in chunks)))
    metadata = SourceMetadata(
        source_id=source_id,
        filename=filename,
        sha256=sha256_hash,
        file_type="json",
        page_count=1,
        char_count=total_chars,
        size=len(file_bytes),
        evidence_locations=locations
    )
    return metadata, chunks


# =====================================================================
# 9. HTML Parser
# =====================================================================
def parse_html(file_bytes: bytes, filename: str) -> Tuple[SourceMetadata, List[SourceChunk]]:
    """Extracts semantic elements and text blocks from HTML documents."""
    sha256_hash = compute_sha256(file_bytes)
    source_id = f"SRC_{sha256_hash[:8]}"
    chunks: List[SourceChunk] = []
    total_chars = 0

    if BeautifulSoup is None:
        return parse_txt(file_bytes, filename)

    try:
        try:
            html_content = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            html_content = file_bytes.decode("latin-1", errors="replace")

        soup = BeautifulSoup(html_content, "html.parser")
        # Remove active, script, style, and embedding elements
        for s in soup(["script", "style", "meta", "noscript", "iframe", "object", "embed", "link", "applet", "svg"]):
            s.extract()

        chunk_idx = 1
        tags_to_extract = ["article", "section", "h1", "h2", "h3", "p", "li", "tr"]

        elements = soup.find_all(tags_to_extract)
        for el in elements:
            text = el.get_text(strip=True)
            if len(text) > 15:
                tag_name = el.name
                class_or_id = el.get("id") or (el.get("class")[0] if el.get("class") else "")
                loc_str = f"<{tag_name}>" + (f" #{class_or_id}" if class_or_id else "")
                total_chars += len(text)

                chunks.append(SourceChunk(
                    source_id=source_id,
                    filename=filename,
                    page=1,
                    chunk_id=f"{source_id}-HTML{chunk_idx:03d}",
                    text=text,
                    start_char=None,
                    end_char=None,
                    file_type="html",
                    location_type="element",
                    location=loc_str
                ))
                chunk_idx += 1

        # Fallback if no specific tags matched
        if not chunks:
            body_text = soup.get_text(separator="\n", strip=True)
            if body_text:
                return parse_txt(body_text.encode("utf-8"), filename)

    except Exception:
        pass

    locations = sorted(list(set(c.location for c in chunks)))
    metadata = SourceMetadata(
        source_id=source_id,
        filename=filename,
        sha256=sha256_hash,
        file_type="html",
        page_count=1,
        char_count=total_chars,
        size=len(file_bytes),
        evidence_locations=locations
    )
    return metadata, chunks


# =====================================================================
# Unified Document Ingestion Router
# =====================================================================
def parse_document(file_bytes: bytes, filename: str) -> Tuple[SourceMetadata, List[SourceChunk]]:
    """
    Unified entry point for format-aware document parsing.
    Validates file extension and content, dispatches to format-specific extractor.
    """
    clean_filename = sanitize_filename(filename)

    # Empty document handling
    if not file_bytes:
        sha256_hash = compute_sha256(b"")
        source_id = f"SRC_{sha256_hash[:8]}"
        meta = SourceMetadata(
            source_id=source_id,
            filename=clean_filename,
            sha256=sha256_hash,
            file_type=clean_filename.split(".")[-1].lower() if "." in clean_filename else "txt",
            page_count=1,
            char_count=0,
            size=0,
            evidence_locations=["Empty document (0 bytes)"]
        )
        return meta, []

    # File type and content validation
    is_valid, err_msg = validate_file_type_and_content(clean_filename, file_bytes)
    if not is_valid:
        log_security_event("INVALID_FILE_UPLOAD", {
            "filename": clean_filename,
            "size": len(file_bytes),
            "reason": err_msg
        })
        sha256_hash = compute_sha256(file_bytes)
        source_id = f"SRC_{sha256_hash[:8]}"
        meta = SourceMetadata(
            source_id=source_id,
            filename=clean_filename,
            sha256=sha256_hash,
            file_type="unsupported",
            page_count=0,
            char_count=0,
            size=len(file_bytes),
            evidence_locations=[f"Unable to process this document. The file may be corrupted or unsupported."]
        )
        return meta, []

    ext = clean_filename.lower().split(".")[-1] if "." in clean_filename else "txt"

    if ext == "pdf":
        return parse_pdf(file_bytes, clean_filename)
    elif ext in ("docx", "doc"):
        return parse_docx(file_bytes, clean_filename)
    elif ext in ("pptx", "ppt"):
        return parse_pptx(file_bytes, clean_filename)
    elif ext in ("xlsx", "xls"):
        return parse_xlsx(file_bytes, clean_filename)
    elif ext == "csv":
        return parse_csv(file_bytes, clean_filename)
    elif ext in ("md", "markdown"):
        return parse_markdown(file_bytes, clean_filename)
    elif ext == "json":
        return parse_json(file_bytes, clean_filename)
    elif ext in ("html", "htm"):
        return parse_html(file_bytes, clean_filename)
    elif ext in ("png", "jpg", "jpeg", "webp"):
        sha256_hash = compute_sha256(file_bytes)
        source_id = f"SRC_{sha256_hash[:8]}"
        meta = SourceMetadata(
            source_id=source_id,
            filename=clean_filename,
            sha256=sha256_hash,
            file_type=ext,
            page_count=1,
            char_count=0,
            size=len(file_bytes),
            evidence_locations=["Image OCR support coming soon"]
        )
        return meta, []
    else:
        # Default text parser
        return parse_txt(file_bytes, clean_filename)

