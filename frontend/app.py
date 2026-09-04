import os
import sys
import json
import requests
import streamlit as st
from datetime import datetime

# Add project root to sys.path to enable direct in-process pipeline calls if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.utils.helpers import SAMPLE_DEMO_AI_DRAFT, SAMPLE_DEMO_SOURCE_TEXT
from backend.services.verifier import VerificationPipeline
from backend.services.certificate import generate_human_readable_report

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="VERDICT: Claim Validato",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished Hackathon UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .code-font {
        font-family: 'JetBrains Mono', monospace;
    }

    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
        color: white;
    }

    .status-badge-verified {
        background-color: #064e3b;
        color: #6ee7b7;
        border: 1px solid #059669;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }

    .status-badge-review {
        background-color: #450a0a;
        color: #fca5a5;
        border: 1px solid #dc2626;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }

    .kpi-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #334155;
        text-align: center;
    }

    .claim-badge-supported {
        background-color: #065f46;
        color: #a7f3d0;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }

    .claim-badge-refuted {
        background-color: #991b1b;
        color: #fecaca;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }

    .claim-badge-unverified {
        background-color: #854d0e;
        color: #fef08a;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }

    .evidence-box {
        background-color: #0f172a;
        border-left: 4px solid #3b82f6;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        font-size: 0.95rem;
    }

    .source-meta-tag {
        background-color: #1e293b;
        color: #94a3b8;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/verified-account.png", width=64)
    st.markdown("## **VERDICT**")
    st.markdown("**Claim Validato**")
    st.markdown("*Team Cygnix*")
    st.caption("Track: Generative AI & Trustworthy Systems")
    st.markdown("---")
    
    st.markdown("### 🎯 Core Philosophy")
    st.info("**'A link is not proof.'**\n\nVerification must be a first-class reproducible output derived from exact evidence spans and cryptographic hashes.")

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Settings")
    top_k = st.slider("Top-K Evidence Candidates", min_value=1, max_value=5, value=3)
    similarity_threshold = st.slider("Similarity Threshold", min_value=0.1, max_value=0.6, value=0.25, step=0.05)

    st.markdown("---")
    # Backend connection test
    try:
        health_resp = requests.get(f"{BACKEND_URL}/health", timeout=1.5)
        if health_resp.status_code == 200:
            st.success("🟢 Backend Connected (FastAPI)")
        else:
            st.warning("🟡 Backend status: Non-200")
    except Exception:
        st.info("ℹ️ Local Engine Active (In-Process)")


# Main Header
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px;">
                ⚖️ VERDICT: Claim Validato
            </h1>
            <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 1.05rem;">
                Evidence-Grounded AI Output Verification & Cryptographic Certification
            </p>
        </div>
        <div style="text-align: right;">
            <span style="background: #334155; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">v1.0 MVP</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# Initialize Session State
if "verification_result" not in st.session_state:
    st.session_state.verification_result = None
if "draft_input" not in st.session_state:
    st.session_state.draft_input = ""
if "selected_claim_id" not in st.session_state:
    st.session_state.selected_claim_id = None
if "demo_mode_active" not in st.session_state:
    st.session_state.demo_mode_active = False


# Quick Demo Mode Toggle
demo_col1, demo_col2 = st.columns([3, 1])
with demo_col1:
    st.markdown("Verify every AI-generated claim against its exact source evidence passages in PDFs and TXT documents.")
with demo_col2:
    if st.button("🚀 Load 1-Click Demo", use_container_width=True):
        st.session_state.draft_input = SAMPLE_DEMO_AI_DRAFT
        st.session_state.demo_mode_active = True
        st.toast("Loaded sample compliance directive & AI draft!", icon="📋")


st.markdown("### 1. Upload & Verify")

col_draft, col_sources = st.columns(2)

with col_draft:
    st.markdown("#### 📝 AI-Generated Draft")
    draft_upload = st.file_uploader("Upload AI Draft as TXT (optional)", type=["txt"], key="draft_uploader")
    
    if draft_upload is not None:
        st.session_state.draft_input = draft_upload.read().decode("utf-8", errors="replace")

    draft_text = st.text_area(
        "AI-generated text or answer to verify:",
        value=st.session_state.draft_input,
        height=220,
        placeholder="Paste your AI-generated response, summary, or claims here..."
    )

with col_sources:
    st.markdown("#### 📚 Ground-Truth Source Documents")
    st.caption("**Supported Formats:** `PDF • DOCX • PPTX • XLSX • CSV • TXT • MD • JSON • HTML`")
    uploaded_sources = st.file_uploader(
        "Upload reference documents:",
        type=["pdf", "docx", "pptx", "xlsx", "csv", "txt", "md", "json", "html", "htm"],
        accept_multiple_files=True,
        key="source_uploader"
    )

    if st.session_state.demo_mode_active and not uploaded_sources:
        st.info("📌 **Demo Documents Attached**: `CRD_Compliance_Directive_2026.txt` (Regulatory Compliance Standard)")

    if uploaded_sources:
        st.caption(f"📁 {len(uploaded_sources)} document(s) ready for ingestion:")
        for u in uploaded_sources:
            ext = u.name.split(".")[-1].upper() if "." in u.name else "TXT"
            st.markdown(f"- `✓ {u.name}` **[{ext}]** ({u.size:,} bytes)")


# Verification Trigger Button
st.markdown("<br>", unsafe_allow_html=True)
verify_clicked = st.button("🔍 VERIFY CLAIMS", type="primary", use_container_width=True)

if verify_clicked:
    final_draft = draft_text.strip()
    if not final_draft:
        st.error("Please provide an AI-generated draft to verify.")
    else:
        # Prepare files
        files_to_send = []
        if uploaded_sources:
            for sf in uploaded_sources:
                files_to_send.append((sf.name, sf.getvalue()))
        elif st.session_state.demo_mode_active:
            files_to_send.append(("CRD_Compliance_Directive_2026.txt", SAMPLE_DEMO_SOURCE_TEXT.encode("utf-8")))

        if not files_to_send:
            st.warning("⚠️ No source documents uploaded. Verification will evaluate claims without grounding.")

        # Progress workflow indicator
        progress_placeholder = st.empty()
        with progress_placeholder.container():
            st.markdown("⏳ **Verification in progress...**")
            p_bar = st.progress(10)
            
            p_bar.progress(25, text="1/5 Reading & hashing source documents...")
            p_bar.progress(45, text="2/5 Extracting page & chunk metadata...")
            p_bar.progress(65, text="3/5 Decomposing draft into atomic claims...")
            p_bar.progress(80, text="4/5 Retrieving BAAI/bge-small-en-v1.5 embeddings...")
            p_bar.progress(95, text="5/5 Running entailment verification & assembling certificate...")

        # Run verification: Try backend API first, fallback to direct in-process pipeline
        cert_data = None
        try:
            multipart_files = []
            for fname, b_data in files_to_send:
                multipart_files.append(("source_files", (fname, b_data, "application/octet-stream")))
            
            resp = requests.post(
                f"{BACKEND_URL}/verify",
                data={"draft_text": final_draft},
                files=multipart_files,
                timeout=45
            )
            if resp.status_code == 200:
                data = resp.json()
                cert_data = data.get("certificate")
        except Exception:
            pass

        if not cert_data:
            # Direct in-process verification fallback
            pipeline = VerificationPipeline()
            cert_obj = pipeline.run_verification(
                draft_text=final_draft,
                source_files=files_to_send,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            cert_data = cert_obj.model_dump()

        progress_placeholder.empty()
        st.session_state.verification_result = cert_data
        st.session_state.selected_claim_id = cert_data["claims"][0]["claim_id"] if cert_data.get("claims") else None
        st.toast("Verification Complete! Certificate Generated.", icon="✅")


# DISPLAY VERIFICATION RESULTS
if st.session_state.verification_result:
    cert = st.session_state.verification_result
    summary = cert.get("summary", {})
    claims = cert.get("claims", [])
    overall_verdict = cert.get("overall_verdict", "REVIEW_REQUIRED")
    cert_id = cert.get("certificate_id", "N/A")

    st.markdown("---")
    st.markdown("### 2. Executive Verification Summary")

    # Overall Verdict Badge
    col_verdict, col_total, col_sup, col_ref, col_unv = st.columns([2, 1, 1, 1, 1])

    with col_verdict:
        if overall_verdict == "VERIFIED":
            st.markdown(f"""
            <div style="background: #064e3b; border: 1px solid #059669; padding: 16px; border-radius: 10px;">
                <span style="font-size: 0.8rem; color: #a7f3d0; text-transform: uppercase; font-weight: 700;">OVERALL STATUS</span>
                <h2 style="margin: 4px 0 0 0; color: #34d399; font-weight: 800;">🟢 VERIFIED</h2>
                <small style="color: #6ee7b7;">All claims grounded in verified evidence.</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: #450a0a; border: 1px solid #dc2626; padding: 16px; border-radius: 10px;">
                <span style="font-size: 0.8rem; color: #fecaca; text-transform: uppercase; font-weight: 700;">OVERALL STATUS</span>
                <h2 style="margin: 4px 0 0 0; color: #f87171; font-weight: 800;">🔴 REVIEW REQUIRED</h2>
                <small style="color: #fca5a5;">One or more claims are refuted or unverified.</small>
            </div>
            """, unsafe_allow_html=True)

    with col_total:
        st.metric("Total Claims", summary.get("total_claims", 0))
    with col_sup:
        st.metric("Supported", summary.get("supported", 0), delta="Evidence Found", delta_color="normal")
    with col_ref:
        st.metric("Refuted", summary.get("refuted", 0), delta="Contradiction", delta_color="inverse")
    with col_unv:
        st.metric("Unverified", summary.get("unverified", 0), delta="Inconclusive", delta_color="off")

    st.markdown("<br>", unsafe_allow_html=True)

    # TWO COLUMN REVIEW INTERFACE: Left = Claim List / Color Coded Draft, Right = Evidence Deep Dive
    st.markdown("### 3. Claim Review & Evidence Deep-Dive")
    st.caption("Click on any atomic claim below to inspect its exact source evidence span, retrieval score, and entailment rationale.")

    col_claims_list, col_evidence_detail = st.columns([1, 1])

    with col_claims_list:
        st.markdown("#### 📋 Extracted Atomic Claims")
        
        for c in claims:
            cid = c["claim_id"]
            verdict = c["verdict"]
            text = c["text"]

            if verdict == "SUPPORTED":
                v_badge = "🟢 SUPPORTED"
                card_style = "border-left: 4px solid #10b981;"
            elif verdict == "REFUTED":
                v_badge = "🔴 REFUTED"
                card_style = "border-left: 4px solid #ef4444;"
            else:
                v_badge = "🟡 UNVERIFIED"
                card_style = "border-left: 4px solid #f59e0b;"

            # Clickable button for each claim
            is_selected = (st.session_state.selected_claim_id == cid)
            btn_label = f"[{cid}] {v_badge} — {text[:55]}..."
            
            if st.button(btn_label, key=f"btn_{cid}", use_container_width=True):
                st.session_state.selected_claim_id = cid
                st.rerun()

    # Find the currently selected claim
    selected_claim = next((c for c in claims if c["claim_id"] == st.session_state.selected_claim_id), claims[0] if claims else None)

    with col_evidence_detail:
        if selected_claim:
            cid = selected_claim["claim_id"]
            verdict = selected_claim["verdict"]
            c_text = selected_claim["text"]
            reason = selected_claim.get("reason", "")
            conf = selected_claim.get("confidence", 0.0)
            retrieval_score = selected_claim.get("retrieval_score", 0.0)
            evidence_list = selected_claim.get("evidence", [])

            st.markdown(f"#### 🔎 Evidence Inspector: `{cid}`")

            # Verdict Status Callout
            if verdict == "SUPPORTED":
                st.success(f"**VERDICT: SUPPORTED** (Confidence: {conf:.2f})")
            elif verdict == "REFUTED":
                st.error(f"**VERDICT: REFUTED** — Review Required! (Confidence: {conf:.2f})")
            else:
                st.warning(f"**VERDICT: UNVERIFIED** — Inconclusive or Missing Evidence (Confidence: {conf:.2f})")

            # AI Claim
            st.markdown(f"**AI Claim:**\n> *\"{c_text}\"*")
            
            # Rationale
            st.markdown(f"**Entailment Rationale:**\n{reason}")

            # Evidence Span Details
            st.markdown("##### 📌 Ground-Truth Evidence Grounding:")
            if evidence_list:
                top_ev = evidence_list[0]
                loc_type = top_ev.get('location_type') or "location"
                loc_val = top_ev.get('location') or (f"Page {top_ev.get('page')}" if top_ev.get('page') else "N/A")
                file_fmt = str(top_ev.get('file_type') or top_ev.get('source', '').split('.')[-1]).upper()

                st.markdown(f"""
                <div class="evidence-box">
                    <p style="margin: 0; font-style: italic; color: #f8fafc;">
                        "{top_ev.get('text')}"
                    </p>
                    <div style="margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap;">
                        <span class="source-meta-tag">📄 Source: {top_ev.get('source')}</span>
                        <span class="source-meta-tag">🏷️ Format: {file_fmt}</span>
                        <span class="source-meta-tag">📍 {loc_type.capitalize()}: {loc_val}</span>
                        <span class="source-meta-tag">🎯 Cosine Similarity: {top_ev.get('similarity', 0.0):.2f}</span>
                        <span class="source-meta-tag">🆔 Chunk: {top_ev.get('chunk_id', 'N/A')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if len(evidence_list) > 1:
                    with st.expander(f"View {len(evidence_list)-1} other candidate passages"):
                        for ev in evidence_list[1:]:
                            ev_loc = ev.get('location') or (f"Page {ev.get('page')}" if ev.get('page') else "")
                            st.markdown(f"- *\"{ev.get('text')}\"*")
                            st.caption(f"Source: `{ev.get('source')}` ({ev_loc}) | Similarity: {ev.get('similarity', 0.0):.2f}")
            else:
                st.info("No matching evidence passages retrieved from source documents.")
        else:
            st.info("Select a claim from the left panel to inspect grounding evidence.")

    # CERTIFICATE & AUDIT TRAIL SECTION
    st.markdown("---")
    st.markdown("### 4. Cryptographic Verification Certificate")
    st.caption("A tamper-evident, machine-readable audit trail anchoring verified claims to source document SHA-256 hashes.")

    cert_col1, cert_col2 = st.columns([2, 1])

    with cert_col1:
        st.markdown(f"**Certificate ID**: `{cert_id}`")
        st.markdown(f"**Verification Timestamp**: `{cert.get('timestamp')}`")
        st.markdown(f"**Embedding Model**: `{cert.get('configuration', {}).get('embedding_model', 'BAAI/bge-small-en-v1.5')}`")
        st.markdown(f"**Entailment Engine**: `{cert.get('configuration', {}).get('entailment_provider', 'heuristic_fallback')}`")

        # Source Document Hashes
        st.markdown("##### 🔐 Source Integrity Audit:")
        sources = cert.get("sources", [])
        if sources:
            source_table = [
                {
                    "Filename": s["filename"],
                    "Format": str(s.get("file_type", "doc")).upper(),
                    "SHA-256 Checksum": f"{s['sha256'][:20]}...",
                    "Evidence Scope": f"{len(s.get('evidence_locations', []))} location(s)" if s.get('evidence_locations') else f"{s.get('page_count', 1)} page(s)"
                }
                for s in sources
            ]
            st.table(source_table)
        else:
            st.caption("No source files hashed.")

    with cert_col2:
        st.markdown("##### 📥 Export Artifacts")
        
        # Download JSON Certificate
        cert_json_str = json.dumps(cert, indent=2)
        st.download_button(
            label="📄 Download JSON Certificate",
            data=cert_json_str,
            file_name=f"{cert_id}.json",
            mime="application/json",
            use_container_width=True
        )

        # Generate & Download Human-Readable Markdown Report
        from backend.schemas import VerificationCertificate
        try:
            cert_obj = VerificationCertificate.model_validate(cert)
            report_md = generate_human_readable_report(cert_obj)
        except Exception:
            report_md = f"# Verification Report\nCertificate ID: {cert_id}\nOverall Verdict: {overall_verdict}"

        st.download_button(
            label="📝 Download Human-Readable Report",
            data=report_md,
            file_name=f"{cert_id}_report.md",
            mime="text/markdown",
            use_container_width=True
        )

    with st.expander("🔍 View Raw JSON Certificate"):
        st.json(cert)

st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("VERDICT: Claim Validato • Team Cygnix • Generative AI & Trustworthy Systems Track • 2026")
