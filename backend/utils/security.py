import os
import re
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger("verdict_security")

# =====================================================================
# Configurable Limits (via Environment Variables)
# =====================================================================
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "25"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

MAX_TOTAL_UPLOAD_SIZE_MB = int(os.getenv("MAX_TOTAL_UPLOAD_SIZE_MB", "50"))
MAX_TOTAL_UPLOAD_SIZE_BYTES = MAX_TOTAL_UPLOAD_SIZE_MB * 1024 * 1024

MAX_DRAFT_CHARS = int(os.getenv("MAX_DRAFT_CHARS", "50000"))
MAX_SOURCE_FILES = int(os.getenv("MAX_SOURCE_FILES", "15"))

# =====================================================================
# Strict Extension Allowlist
# =====================================================================
ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".csv",
    ".md",
    ".markdown",
    ".json",
    ".html",
    ".htm",
    ".txt"
}

# Explicitly rejected dangerous extensions
DISALLOWED_EXTENSIONS = {
    ".exe", ".bat", ".cmd", ".sh", ".py", ".php", ".js", ".vbs",
    ".ps1", ".jar", ".bin", ".dll", ".so", ".dylib", ".msi", ".scr",
    ".cgi", ".pl", ".rb", ".elf", ".com", ".pif", ".c", ".cpp"
}


# =====================================================================
# Filename Sanitization & Path Traversal Prevention
# =====================================================================
def sanitize_filename(raw_name: Optional[str]) -> str:
    r"""
    Sanitizes user-provided filename:
    - Strips directory components (e.g. ../, ..\, /etc/passwd)
    - Strips null bytes and non-printable control characters
    - Removes leading periods to prevent hidden file tricks
    - Replaces dangerous characters with underscores
    - Limits length to 255 characters
    """
    if not raw_name or not raw_name.strip():
        return "unnamed_document.txt"

    # Normalize backslashes to forward slashes first to handle Windows paths across OSes
    normalized = str(raw_name).replace('\\', '/')

    # Extract base name only (strips path components)
    clean = Path(normalized).name

    # Remove null bytes and control characters
    clean = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', clean)

    # Remove dangerous characters like path separators and shell characters
    clean = re.sub(r'[\\/:*?"<>|;`$]', '_', clean)

    # Strip any remaining '..' sequences
    clean = clean.replace('..', '')

    # Strip leading dots and spaces to avoid hidden files
    clean = clean.lstrip('. ')

    # Limit filename length
    if len(clean) > 255:
        stem = Path(clean).stem[:240]
        suffix = Path(clean).suffix[:14]
        clean = f"{stem}{suffix}"

    return clean if clean else "unnamed_document.txt"



def safe_resolve_path(base_dir: str, user_filename: str) -> Path:
    """
    Safely resolves a path inside base_dir.
    Raises ValueError if the path attempts directory traversal outside base_dir.
    """
    base = Path(base_dir).resolve()
    # Sanitize the input filename
    safe_name = sanitize_filename(user_filename)
    target = (base / safe_name).resolve()

    if not str(target).startswith(str(base)):
        raise ValueError(f"Path traversal detected: {user_filename}")

    return target


# =====================================================================
# File Extension & Magic Byte / Content Sniffing
# =====================================================================
def validate_file_type_and_content(filename: str, content_bytes: bytes) -> Tuple[bool, str]:
    """
    Validates file extension against strict allowlist and performs content sniffing
    to prevent file extension spoofing (e.g. renaming an .exe or binary to .pdf).
    Returns (is_valid: bool, error_message: str).
    """
    if not filename:
        return False, "Filename cannot be empty."

    # Prevent path traversal tricks in filename
    if ".." in filename or "/" in filename or "\\" in filename:
        clean_name = sanitize_filename(filename)
    else:
        clean_name = filename

    ext = Path(clean_name).suffix.lower()

    if not ext:
        # If no extension, check if it's plain text
        if is_valid_text_content(content_bytes):
            return True, ""
        return False, "File lacks an extension and does not appear to be valid plain text."

    if ext in DISALLOWED_EXTENSIONS:
        return False, f"File extension '{ext}' is explicitly prohibited."

    if ext not in ALLOWED_EXTENSIONS:
        return False, f"File extension '{ext}' is not supported. Supported extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}"

    # Check for empty file
    if len(content_bytes) == 0:
        return False, "File is empty (0 bytes)."

    # Content sniffing by format
    if ext == ".pdf":
        # PDF magic bytes: %PDF- (allow small offset for BOM/whitespace)
        header = content_bytes[:32]
        if b"%PDF-" not in header:
            return False, "Invalid PDF content: Missing standard %PDF header magic bytes."

    elif ext in (".docx", ".pptx", ".xlsx"):
        # Modern Office formats are ZIP archives starting with PK\x03\x04 or PK\x05\x06
        if not (content_bytes.startswith(b"PK\x03\x04") or content_bytes.startswith(b"PK\x05\x06")):
            return False, f"Invalid {ext.upper()[1:]} content: Not a valid OpenXML/ZIP archive container."

    elif ext == ".json":
        # Validate that content is valid JSON
        try:
            parsed = json.loads(content_bytes.decode("utf-8", errors="replace"))
            if not isinstance(parsed, (dict, list)):
                return False, "JSON content must be an object or array at root level."
        except Exception:
            return False, "Invalid JSON content: Unable to parse document as valid JSON."

    elif ext in (".html", ".htm"):
        # Ensure not arbitrary binary masquerading as HTML
        if b"\x00" in content_bytes[:4096]:
            return False, "Invalid HTML content: Binary null bytes detected."

    elif ext in (".csv", ".md", ".markdown", ".txt"):
        # Check for binary executable signatures masquerading as text
        if content_bytes.startswith(b"MZ") or content_bytes.startswith(b"\x7fELF") or content_bytes.startswith(b"\xca\xfe\xba\xbe"):
            return False, "Binary executable signature detected in text file."
        if not is_valid_text_content(content_bytes):
            return False, f"Invalid {ext.upper()[1:]} content: Contains non-text binary data."

    return True, ""


def is_valid_text_content(content_bytes: bytes) -> bool:
    """Checks whether byte sequence is decodeable text without null bytes."""
    if b"\x00" in content_bytes:
        return False
    try:
        content_bytes.decode("utf-8")
        return True
    except UnicodeDecodeError:
        try:
            content_bytes.decode("latin-1")
            return True
        except UnicodeDecodeError:
            return False


# =====================================================================
# Upload Limits & Resource Checks
# =====================================================================
def validate_upload_limits(
    files: List[Tuple[str, bytes]],
    draft_text: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Validates per-file sizes, aggregate size, file count, and draft length.
    Returns (is_valid: bool, error_message: str).
    """
    # 1. Draft text length limit
    if draft_text and len(draft_text) > MAX_DRAFT_CHARS:
        return False, f"AI draft exceeds the maximum allowed length ({MAX_DRAFT_CHARS:,} characters)."

    # 2. Maximum source file count
    if len(files) > MAX_SOURCE_FILES:
        return False, f"Too many source documents uploaded ({len(files)}). Maximum allowed is {MAX_SOURCE_FILES}."

    # 3. Per-file and aggregate size limits
    total_bytes = 0
    for fname, fbytes in files:
        fsize = len(fbytes)
        total_bytes += fsize

        if fsize > MAX_FILE_SIZE_BYTES:
            safe_name = sanitize_filename(fname)
            return False, f"File '{safe_name}' exceeds the maximum allowed size ({MAX_FILE_SIZE_MB} MB)."

    if total_bytes > MAX_TOTAL_UPLOAD_SIZE_BYTES:
        return False, f"Total uploaded documents size ({total_bytes / (1024*1024):.1f} MB) exceeds aggregate limit of {MAX_TOTAL_UPLOAD_SIZE_MB} MB."

    return True, ""


# =====================================================================
# Safe Security Event Logging
# =====================================================================
def log_security_event(event_type: str, metadata: Dict[str, Any]):
    """
    Logs security-relevant events with sanitized metadata.
    Never logs API keys, passwords, full drafts, or sensitive document text.
    """
    safe_meta = {}
    forbidden_keys = {"api_key", "key", "secret", "password", "token", "auth", "draft_text", "content"}

    for k, v in metadata.items():
        if k.lower() in forbidden_keys:
            continue
        if isinstance(v, (str, int, float, bool, list, dict)):
            safe_meta[k] = v

    logger.warning(
        f"[SECURITY_EVENT] type={event_type} " + " ".join(f"{k}={v}" for k, v in safe_meta.items())
    )
