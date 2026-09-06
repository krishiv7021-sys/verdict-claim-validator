import os
import io
import re
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from backend.schemas import (
    VerifyResponse,
    HealthResponse,
    VerificationCertificate,
    normalize_authority_level
)
from backend.services.verifier import VerificationPipeline
from backend.services.certificate import (
    load_certificate_from_disk,
    generate_human_readable_report
)
from backend.utils.helpers import get_demo_package
from backend.utils.security import (
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    MAX_TOTAL_UPLOAD_SIZE_BYTES,
    MAX_TOTAL_UPLOAD_SIZE_MB,
    MAX_DRAFT_CHARS,
    MAX_SOURCE_FILES,
    ALLOWED_EXTENSIONS,
    DISALLOWED_EXTENSIONS,
    sanitize_filename,
    validate_file_type_and_content,
    log_security_event
)
from backend.utils.rate_limiter import rate_limiter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verdict_api")

app = FastAPI(
    title="VERDICT — Claim Verification & Evidence Analysis API",
    description="VERDICT is an evidence-grounded claim verification system that analyzes AI-generated content against trusted source documents, identifies supported, refuted, and unverified claims, and provides precise evidence for each verification decision.",
    version="1.0.0"
)

# Secure, environment-aware CORS configuration
allowed_origins_raw = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8501,http://127.0.0.1:8501,http://localhost:3000"
)
allowed_origins = [orig.strip() for orig in allowed_origins_raw.split(",") if orig.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = VerificationPipeline()


# =====================================================================
# Safe Global Exception Handlers (Zero Traceback Leakage)
# =====================================================================
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """Returns clean JSON without leaking server details."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catches unhandled errors, logs technical trace server-side, returns safe generic message."""
    logger.error(f"Unhandled server error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Verification could not be completed."}
    )


# =====================================================================
# API Endpoints
# =====================================================================
@app.get("/")
def read_root():
    return {
        "project": "VERDICT",
        "subtitle": "Claim Verification & Evidence Analysis",
        "description": "VERDICT is an evidence-grounded claim verification system that analyzes AI-generated content against trusted source documents, identifies supported, refuted, and unverified claims, and provides precise evidence for each verification decision.",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "verify": "POST /verify",
            "certificate": "/certificate/{certificate_id}",
            "report": "/certificate/{certificate_id}/report",
            "demo": "/demo",
            "evaluation": "/evaluation"
        }
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Returns safe component health status without revealing keys or internal paths."""
    provider = os.getenv("LLM_PROVIDER", "heuristic_fallback")
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        embedding_model="BAAI/bge-small-en-v1.5",
        entailment_provider=provider,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.get("/demo")
def get_demo_data():
    """Returns bundled demo draft and source document for instant 1-click demonstrations."""
    return get_demo_package()


@app.get("/evaluation")
def get_evaluation(fresh: bool = False):
    """
    Returns benchmark evaluation metrics including accuracy, precision, recall, F1,
    confusion matrix, and execution timing metrics.
    """
    try:
        from evaluation.evaluate import get_or_run_benchmark
        return get_or_run_benchmark(fresh=fresh)
    except Exception as e:
        logger.error(f"Evaluation benchmark error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Evaluation could not be completed.")


@app.post("/verify", response_model=VerifyResponse)
async def verify_claims_endpoint(
    request: Request,
    draft_text: Optional[str] = Form(None),
    draft_file: Optional[UploadFile] = File(None),
    source_files: List[UploadFile] = File(None),
    source_authorities: Optional[str] = Form(None)
):
    """
    Accepts:
    - AI-generated draft as text or uploaded TXT file
    - One or more source documents (PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, JSON, HTML)
    - Optional source_authorities mapping (JSON string or dict)
    Applies:
    - Rate limiting per client IP
    - Strict file size & aggregate upload limits
    - Allowed extensions allowlist and magic byte validation
    - Filename sanitization & path traversal protection
    - Source authority allowlist validation and weight clamping
    """
    # 1. Rate Limiting Check
    client_ip = request.client.host if request.client else "127.0.0.1"
    allowed, retry_after = rate_limiter.is_allowed(client_ip)
    if not allowed:
        log_security_event("RATE_LIMIT_EXCEEDED", {"ip": client_ip, "retry_after": retry_after})
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Please wait {retry_after} seconds before submitting more verification requests."
        )

    try:
        # 2. Resolve & Validate Draft Text
        resolved_draft = ""
        if draft_file is not None:
            # Enforce size limit on draft file
            d_bytes = await draft_file.read()
            if len(d_bytes) > MAX_FILE_SIZE_BYTES:
                raise HTTPException(status_code=400, detail="File exceeds the maximum allowed size.")
            resolved_draft = d_bytes.decode("utf-8", errors="replace")
        elif draft_text is not None and draft_text.strip():
            resolved_draft = draft_text.strip()

        # Enforce maximum character limit on draft text
        if len(resolved_draft) > MAX_DRAFT_CHARS:
            log_security_event("OVERSIZED_DRAFT_REJECTED", {"chars": len(resolved_draft), "max": MAX_DRAFT_CHARS})
            raise HTTPException(
                status_code=400,
                detail=f"AI draft exceeds the maximum allowed length ({MAX_DRAFT_CHARS:,} characters)."
            )

        if not resolved_draft:
            # Handle empty draft gracefully without error
            pipeline_inst = VerificationPipeline()
            cert = pipeline_inst.run_verification(draft_text="", source_files=[])
            return VerifyResponse(
                status="success",
                certificate=cert,
                draft_text="",
                error="Draft text was empty. No claims to evaluate."
            )

        # 3. Validate Source Document Count
        if source_files and len(source_files) > MAX_SOURCE_FILES:
            log_security_event("EXCESSIVE_SOURCE_COUNT", {"count": len(source_files), "max": MAX_SOURCE_FILES})
            raise HTTPException(
                status_code=400,
                detail=f"Too many source documents uploaded ({len(source_files)}). Maximum allowed is {MAX_SOURCE_FILES}."
            )

        # 4. Parse & Validate Source Authorities Allowlist
        parsed_authorities = {}
        if source_authorities:
            try:
                import json
                if isinstance(source_authorities, str):
                    raw_dict = json.loads(source_authorities)
                elif isinstance(source_authorities, dict):
                    raw_dict = source_authorities
                else:
                    raw_dict = {}

                # Sanitize authority levels against known allowlist and clamp weights
                for fname, raw_auth in raw_dict.items():
                    safe_fname = sanitize_filename(fname)
                    norm_level, norm_weight = normalize_authority_level(str(raw_auth))
                    parsed_authorities[safe_fname] = norm_level
            except Exception as parse_err:
                logger.warning(f"Could not parse source_authorities JSON: {parse_err}")

        # 5. Process & Validate Source Files
        files_to_process = []
        total_upload_bytes = 0

        if source_files:
            for sf in source_files:
                orig_filename = sf.filename or "document.txt"
                safe_filename = sanitize_filename(orig_filename)

                # Check extension against strict allowlist
                ext = Path(safe_filename).suffix.lower()
                if ext in DISALLOWED_EXTENSIONS or (ext and ext not in ALLOWED_EXTENSIONS):
                    log_security_event("REJECTED_FILE_TYPE", {"filename": safe_filename, "ext": ext})
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid file type or unsupported content format for '{safe_filename}'."
                    )

                b = await sf.read()
                file_size = len(b)
                total_upload_bytes += file_size

                # Check per-file size
                if file_size > MAX_FILE_SIZE_BYTES:
                    log_security_event("OVERSIZED_FILE_REJECTED", {"filename": safe_filename, "size": file_size})
                    raise HTTPException(
                        status_code=400,
                        detail="File exceeds the maximum allowed size."
                    )

                # Check total upload size
                if total_upload_bytes > MAX_TOTAL_UPLOAD_SIZE_BYTES:
                    log_security_event("OVERSIZED_TOTAL_UPLOAD_REJECTED", {"total_size": total_upload_bytes})
                    raise HTTPException(
                        status_code=400,
                        detail=f"Total uploaded source files exceed the maximum allowed size ({MAX_TOTAL_UPLOAD_SIZE_MB} MB)."
                    )

                # Validate magic bytes / content structure
                is_valid, content_err = validate_file_type_and_content(safe_filename, b)
                if not is_valid:
                    log_security_event("INVALID_FILE_CONTENT", {"filename": safe_filename, "reason": content_err})
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unable to process this document. {content_err}"
                    )

                files_to_process.append((safe_filename, b))

        # 6. Run Verification Pipeline
        certificate = pipeline.run_verification(
            draft_text=resolved_draft,
            source_files=files_to_process,
            source_authorities=parsed_authorities if parsed_authorities else None,
            top_k=3,
            similarity_threshold=0.25
        )

        return VerifyResponse(
            status="success",
            certificate=certificate,
            draft_text=resolved_draft
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verification error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Verification could not be completed due to an internal processing error."
        )


@app.get("/certificate/{certificate_id}", response_model=VerificationCertificate)
def get_certificate(certificate_id: str):
    """Retrieves a previously generated JSON certificate from disk with path traversal protection."""
    if not re.match(r'^[a-zA-Z0-9_-]{1,64}$', certificate_id):
        raise HTTPException(status_code=400, detail="Invalid certificate identifier.")

    try:
        return load_certificate_from_disk(certificate_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Certificate '{certificate_id}' not found.")
    except Exception as e:
        logger.error(f"Error loading certificate {certificate_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unable to retrieve certificate.")


@app.get("/certificate/{certificate_id}/report", response_class=PlainTextResponse)
def get_certificate_report(certificate_id: str):
    """Exports human-readable Markdown verification report with path traversal protection."""
    if not re.match(r'^[a-zA-Z0-9_-]{1,64}$', certificate_id):
        raise HTTPException(status_code=400, detail="Invalid certificate identifier.")

    try:
        cert = load_certificate_from_disk(certificate_id)
        return generate_human_readable_report(cert)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Certificate '{certificate_id}' not found.")
    except Exception as e:
        logger.error(f"Error generating report for {certificate_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unable to generate certificate report.")


@app.post("/certificate/{certificate_id}/export")
def export_certificate(certificate_id: str):
    """Exports certificate data as downloadable JSON with path traversal protection."""
    if not re.match(r'^[a-zA-Z0-9_-]{1,64}$', certificate_id):
        raise HTTPException(status_code=400, detail="Invalid certificate identifier.")

    try:
        cert = load_certificate_from_disk(certificate_id)
        return cert
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Certificate '{certificate_id}' not found.")
