import os
import io
import logging
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from backend.schemas import (
    VerifyResponse,
    HealthResponse,
    VerificationCertificate
)
from backend.services.verifier import VerificationPipeline
from backend.services.certificate import (
    load_certificate_from_disk,
    generate_human_readable_report
)
from backend.utils.helpers import get_demo_package

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verdict_api")

app = FastAPI(
    title="VERDICT: Claim Validato API",
    description="A reproducible, evidence-grounded verification layer that ties every AI-generated claim to its exact source text.",
    version="1.0.0"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = VerificationPipeline()


@app.get("/")
def read_root():
    return {
        "project": "VERDICT: Claim Validato",
        "team": "Cygnix",
        "track": "Generative AI & Trustworthy Systems",
        "tagline": "A reproducible, evidence-grounded verification layer that ties every AI-generated claim to its exact source text.",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "verify": "POST /verify",
            "certificate": "/certificate/{certificate_id}",
            "report": "/certificate/{certificate_id}/report",
            "demo": "/demo"
        }
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
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


@app.post("/verify", response_model=VerifyResponse)
async def verify_claims_endpoint(
    draft_text: Optional[str] = Form(None),
    draft_file: Optional[UploadFile] = File(None),
    source_files: List[UploadFile] = File(None)
):
    """
    Accepts:
    - AI-generated draft as text or uploaded TXT file
    - One or more source documents (PDF or TXT)
    Returns:
    - Complete verification certificate with individual claim results, evidence spans, and cryptographic hashes.
    """
    try:
        # Resolve draft text
        resolved_draft = ""
        if draft_file is not None:
            content_bytes = await draft_file.read()
            resolved_draft = content_bytes.decode("utf-8", errors="replace")
        elif draft_text is not None and draft_text.strip():
            resolved_draft = draft_text.strip()

        if not resolved_draft:
            # Handle empty draft gracefully without 500 error
            pipeline_inst = VerificationPipeline()
            cert = pipeline_inst.run_verification(
                draft_text="",
                source_files=[]
            )
            return VerifyResponse(
                status="success",
                certificate=cert,
                draft_text="",
                error="Draft text was empty. No claims to evaluate."
            )

        # Process source documents
        files_to_process = []
        if source_files:
            for sf in source_files:
                filename = sf.filename or "document.txt"
                b = await sf.read()
                if b:
                    files_to_process.append((filename, b))

        # Run verification pipeline
        certificate = pipeline.run_verification(
            draft_text=resolved_draft,
            source_files=files_to_process,
            top_k=3,
            similarity_threshold=0.25
        )

        return VerifyResponse(
            status="success",
            certificate=certificate,
            draft_text=resolved_draft
        )

    except Exception as e:
        logger.error(f"Verification error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Verification failed: {str(e)}"
        )


@app.get("/certificate/{certificate_id}", response_model=VerificationCertificate)
def get_certificate(certificate_id: str):
    """Retrieves a previously generated JSON certificate from disk."""
    try:
        return load_certificate_from_disk(certificate_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Certificate '{certificate_id}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/certificate/{certificate_id}/report", response_class=PlainTextResponse)
def get_certificate_report(certificate_id: str):
    """Exports human-readable Markdown verification report."""
    try:
        cert = load_certificate_from_disk(certificate_id)
        return generate_human_readable_report(cert)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Certificate '{certificate_id}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/certificate/{certificate_id}/export")
def export_certificate(certificate_id: str):
    """Exports certificate data as downloadable JSON."""
    try:
        cert = load_certificate_from_disk(certificate_id)
        return cert
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Certificate '{certificate_id}' not found.")
