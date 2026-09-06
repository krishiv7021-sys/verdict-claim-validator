import os
import sys
import json
import html
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
from backend.schemas import (
    SourceAuthorityLevel,
    AUTHORITY_WEIGHTS,
    DEFAULT_AUTHORITY_LEVEL,
    VerificationCertificate
)
from evaluation.evaluate import get_or_run_benchmark, run_benchmark

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="VERDICT — Claim Verification & Evidence Analysis",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished professional UI
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

    .conflict-evidence-box {
        background-color: #1a0f0f;
        border-left: 4px solid #ef4444;
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

    .authority-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.3px;
    }

    .auth-statutory {
        background: #1e1b4b;
        color: #a5b4fc;
        border: 1px solid #4338ca;
    }

    .auth-policy {
        background: #082f49;
        color: #7dd3fc;
        border: 1px solid #0284c7;
    }

    .auth-internal {
        background: #1e293b;
        color: #cbd5e1;
        border: 1px solid #475569;
    }

    .auth-reference {
        background: #27272a;
        color: #d4d4d8;
        border: 1px solid #52525b;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/verified-account.png", width=64)
    st.markdown("## **VERDICT**")
    st.markdown("**Claim Verification & Evidence Analysis**")
    st.caption("Precision AI Content Verification Engine")
    st.markdown("---")
    
    st.markdown("### 🎯 Core Philosophy")
    st.info("**'A link is not proof.'**\n\nVERDICT verifies AI-generated claims against source documents and provides precise, inspectable evidence for every decision.")

    st.markdown("---")
    st.markdown("### 🏛️ Source Authority Levels")
    st.markdown("""
    - **Statutory / Official** (1.00): Regulatory acts, statutes, legal decrees
    - **Policy / Regulation** (0.97): Organizational policies, standards
    - **Internal / Org** (0.94): Internal memos, procedures, runbooks
    - **Reference / Web** (0.90): Third-party guides, docs, general reference
    """)

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
                ⚖️ VERDICT
            </h1>
            <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 1.05rem; font-weight: 500;">
                Claim Verification & Evidence Analysis
            </p>
        </div>
        <div style="text-align: right;">
            <span style="background: #334155; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">v1.1 Advanced</span>
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
if "source_authorities_selection" not in st.session_state:
    st.session_state.source_authorities_selection = {}
if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = None

# TOP LEVEL TABS: 1) Verification & Evidence Analysis, 2) Evaluation Dashboard
tab_verify, tab_eval = st.tabs(["🔍 Verify Claims & Evidence", "📊 Evaluation Dashboard & Benchmarks"])

with tab_verify:
    # Quick Demo Mode Toggle
    demo_col1, demo_col2 = st.columns([3, 1])
    with demo_col1:
        st.markdown("VERDICT is an evidence-grounded claim verification system that analyzes AI-generated content against trusted source documents, identifies supported, refuted, and unverified claims, and provides precise evidence for each verification decision.")
    with demo_col2:
        if st.button("🚀 Load 1-Click Demo", use_container_width=True):
            st.session_state.draft_input = SAMPLE_DEMO_AI_DRAFT
            st.session_state.demo_mode_active = True
            st.toast("Loaded sample compliance directive & AI draft!", icon="📋")

    st.markdown("### 1. Upload & Configure Sources")

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

        authority_options = [
            SourceAuthorityLevel.INTERNAL.value,
            SourceAuthorityLevel.STATUTORY.value,
            SourceAuthorityLevel.POLICY.value,
            SourceAuthorityLevel.REFERENCE.value
        ]

        if st.session_state.demo_mode_active and not uploaded_sources:
            st.info("📌 **Demo Documents Attached**: `CRD_Compliance_Directive.txt` (Regulatory Compliance Standard)")
            st.session_state.source_authorities_selection["CRD_Compliance_Directive.txt"] = SourceAuthorityLevel.POLICY.value

        if uploaded_sources:
            st.caption(f"📁 {len(uploaded_sources)} document(s) ready for ingestion — configure authority levels:")
            for u in uploaded_sources:
                ext = u.name.split(".")[-1].upper() if "." in u.name else "TXT"
                current_val = st.session_state.source_authorities_selection.get(u.name, SourceAuthorityLevel.INTERNAL.value)
                c_f1, c_f2 = st.columns([3, 2])
                with c_f1:
                    st.markdown(f"`✓ {u.name}` **[{ext}]** ({u.size:,} B)")
                with c_f2:
                    sel = st.selectbox(
                        f"Authority for {u.name}",
                        options=authority_options,
                        index=authority_options.index(current_val) if current_val in authority_options else 0,
                        key=f"auth_{u.name}",
                        label_visibility="collapsed"
                    )
                    st.session_state.source_authorities_selection[u.name] = sel

    # Verification Trigger Button
    st.markdown("<br>", unsafe_allow_html=True)
    verify_clicked = st.button("🔍 VERIFY CLAIMS", type="primary", use_container_width=True)

    if verify_clicked:
        final_draft = draft_text.strip()
        if not final_draft:
            st.error("Please provide an AI-generated draft to verify.")
        elif len(final_draft) > 50000:
            st.error(f"AI draft exceeds maximum allowed length of 50,000 characters (current: {len(final_draft):,}).")
        else:
            # Validate source file sizes before submission
            oversized = False
            total_source_bytes = 0
            if uploaded_sources:
                for sf in uploaded_sources:
                    total_source_bytes += sf.size
                    if sf.size > 25 * 1024 * 1024:
                        st.error(f"File '{sf.name}' exceeds the maximum allowed size of 25 MB.")
                        oversized = True
                        break
                if total_source_bytes > 50 * 1024 * 1024:
                    st.error("Total uploaded files exceed the aggregate limit of 50 MB.")
                    oversized = True

            if not oversized:
                # Prepare files
                files_to_send = []
                if uploaded_sources:
                    for sf in uploaded_sources:
                        files_to_send.append((sf.name, sf.getvalue()))
                elif st.session_state.demo_mode_active:
                    files_to_send.append(("CRD_Compliance_Directive.txt", SAMPLE_DEMO_SOURCE_TEXT.encode("utf-8")))

                if not files_to_send:
                    st.warning("⚠️ No source documents uploaded. Verification will evaluate claims without grounding.")

                # Progress workflow indicator
                progress_placeholder = st.empty()
                with progress_placeholder.container():
                    st.markdown("⏳ **Verification in progress...**")
                    p_bar = st.progress(10)
                    
                    p_bar.progress(25, text="1/5 Reading & hashing source documents...")
                    p_bar.progress(45, text="2/5 Extracting page & chunk metadata + authority weighting...")
                    p_bar.progress(65, text="3/5 Decomposing draft into atomic claims...")
                    p_bar.progress(80, text="4/5 Retrieving BAAI/bge-small-en-v1.5 embeddings & conflict analysis...")
                    p_bar.progress(95, text="5/5 Running entailment verification & assembling certificate...")

                # Run verification: Try backend API first, fallback to direct in-process pipeline
                cert_data = None
                authorities_payload = st.session_state.source_authorities_selection
                try:
                    multipart_files = []
                    for fname, b_data in files_to_send:
                        multipart_files.append(("source_files", (fname, b_data, "application/octet-stream")))
                    
                    form_data = {
                        "draft_text": final_draft,
                        "source_authorities": json.dumps(authorities_payload)
                    }

                    resp = requests.post(
                        f"{BACKEND_URL}/verify",
                        data=form_data,
                        files=multipart_files,
                        timeout=45
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        cert_data = data.get("certificate")
                    else:
                        try:
                            err_detail = resp.json().get("detail", f"Verification could not be completed (HTTP {resp.status_code}).")
                        except Exception:
                            err_detail = f"Verification could not be completed (HTTP {resp.status_code})."
                        progress_placeholder.empty()
                        st.error(f"❌ {err_detail}")
                        cert_data = None
                except requests.exceptions.ConnectionError:
                    # Backend server not running - local in-process fallback
                    pipeline = VerificationPipeline()
                    cert_obj = pipeline.run_verification(
                        draft_text=final_draft,
                        source_files=files_to_send,
                        source_authorities=authorities_payload if authorities_payload else None,
                        top_k=top_k,
                        similarity_threshold=similarity_threshold
                    )
                    cert_data = cert_obj.model_dump()
                except Exception as ex:
                    progress_placeholder.empty()
                    st.error(f"❌ Verification request error: {ex}")
                    cert_data = None

                progress_placeholder.empty()
                if cert_data:
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
        conflicts_count = summary.get("conflicts_detected", 0)

        st.markdown("---")
        st.markdown("### 2. Executive Verification Summary")

        # Overall Verdict Badge & KPI Metrics
        col_verdict, col_total, col_sup, col_ref, col_unv, col_conf = st.columns([2, 1, 1, 1, 1, 1])

        with col_verdict:
            if overall_verdict == "VERIFIED":
                st.markdown("""
                <div style="background: #064e3b; border: 1px solid #059669; padding: 16px; border-radius: 10px;">
                    <span style="font-size: 0.8rem; color: #a7f3d0; text-transform: uppercase; font-weight: 700;">OVERALL STATUS</span>
                    <h2 style="margin: 4px 0 0 0; color: #34d399; font-weight: 800;">🟢 VERIFIED</h2>
                    <small style="color: #6ee7b7;">All claims grounded in verified evidence.</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                conflict_note = "Conflicting evidence detected across sources." if conflicts_count > 0 else "One or more claims are refuted or unverified."
                st.markdown(f"""
                <div style="background: #450a0a; border: 1px solid #dc2626; padding: 16px; border-radius: 10px;">
                    <span style="font-size: 0.8rem; color: #fecaca; text-transform: uppercase; font-weight: 700;">OVERALL STATUS</span>
                    <h2 style="margin: 4px 0 0 0; color: #f87171; font-weight: 800;">🔴 REVIEW REQUIRED</h2>
                    <small style="color: #fca5a5;">{conflict_note}</small>
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
        with col_conf:
            st.metric(
                "Source Conflicts",
                conflicts_count,
                delta=f"{conflicts_count} Conflict(s)" if conflicts_count > 0 else "None",
                delta_color="inverse" if conflicts_count > 0 else "normal"
            )

        # Performance banner
        tot_time = summary.get("total_time_seconds", 0.0)
        avg_ms = summary.get("avg_time_per_claim_ms", 0.0)
        st.caption(f"⚡ **Verification Performance**: Total Pipeline Time: **{tot_time:.3f}s** | Average Claim Latency: **{avg_ms:.1f}ms**")

        st.markdown("<br>", unsafe_allow_html=True)

        # TWO COLUMN REVIEW INTERFACE
        st.markdown("### 3. Claim Review & Evidence Deep-Dive")
        st.caption("Click on any atomic claim below to inspect its exact source evidence span, retrieval score, and entailment rationale.")

        col_claims_list, col_evidence_detail = st.columns([1, 1])

        with col_claims_list:
            st.markdown("#### 📋 Extracted Atomic Claims")
            
            for c in claims:
                cid = c["claim_id"]
                verdict = c["verdict"]
                text = c.get("claim_text") or c["text"]
                has_conflict = c.get("conflict_detected", False)

                if has_conflict:
                    v_badge = "⚠️ CONFLICT"
                elif verdict == "SUPPORTED":
                    v_badge = "🟢 SUPPORTED"
                elif verdict == "REFUTED":
                    v_badge = "🔴 REFUTED"
                else:
                    v_badge = "🟡 UNVERIFIED"

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
                c_text = selected_claim.get("claim_text") or selected_claim["text"]
                reason = selected_claim.get("explanation") or selected_claim.get("reason", "")
                conf = selected_claim.get("confidence", 0.0)
                sim_score = selected_claim.get("similarity_score") or selected_claim.get("retrieval_score", 0.0)
                rank_score = selected_claim.get("ranking_score", sim_score)
                src_auth = selected_claim.get("source_authority", DEFAULT_AUTHORITY_LEVEL)
                has_conflict = selected_claim.get("conflict_detected", False)
                claim_ms = selected_claim.get("processing_time_ms", 0.0)

                st.markdown(f"#### 🔎 Evidence Inspector: `{cid}`")

                # Verdict Status Callout
                if has_conflict:
                    st.error(f"**VERDICT: REVIEW REQUIRED — CONFLICT DETECTED** (Confidence: {conf:.2f} | Latency: {claim_ms:.1f}ms)")
                elif verdict == "SUPPORTED":
                    st.success(f"**VERDICT: SUPPORTED** (Confidence: {conf:.2f} | Latency: {claim_ms:.1f}ms)")
                elif verdict == "REFUTED":
                    st.error(f"**VERDICT: REFUTED** — Review Required! (Confidence: {conf:.2f} | Latency: {claim_ms:.1f}ms)")
                else:
                    st.warning(f"**VERDICT: UNVERIFIED** — Inconclusive or Missing Evidence (Confidence: {conf:.2f} | Latency: {claim_ms:.1f}ms)")

                # AI Claim
                st.markdown(f"**AI Claim:**\n> *\"{c_text}\"*")
                
                # Rationale
                st.markdown(f"**Entailment Rationale:**\n{reason}")

                # Authority Helper Badge
                def get_auth_badge_html(auth_str):
                    if "STATUTORY" in auth_str:
                        return f'<span class="authority-badge auth-statutory">🏛️ STATUTORY [1.00]</span>'
                    elif "POLICY" in auth_str:
                        return f'<span class="authority-badge auth-policy">📜 POLICY [0.97]</span>'
                    elif "REFERENCE" in auth_str:
                        return f'<span class="authority-badge auth-reference">🌐 REFERENCE [0.90]</span>'
                    else:
                        return f'<span class="authority-badge auth-internal">🏢 INTERNAL [0.94]</span>'

                # If Conflict Detected, Show Conflict Alert & Primary vs Conflicting Evidence
                if has_conflict:
                    st.markdown("""
                    <div style="background: #450a0a; border: 1px solid #dc2626; padding: 12px 16px; border-radius: 8px; margin: 12px 0;">
                        <h4 style="margin: 0; color: #fca5a5; font-size: 0.95rem;">⚠️ Conflicting Evidence Detected Across Sources</h4>
                        <p style="margin: 4px 0 0 0; color: #fecaca; font-size: 0.85rem;">
                            Multiple ground-truth documents contain contradictory facts regarding this claim. Human review is required.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    primary_ev = selected_claim.get("primary_evidence")
                    conflicting_evs = selected_claim.get("conflicting_evidence", [])

                    if primary_ev:
                        p_auth = primary_ev.get("authority_level", DEFAULT_AUTHORITY_LEVEL)
                        p_text_esc = html.escape(str(primary_ev.get('text', '')))
                        p_src_esc = html.escape(str(primary_ev.get('source', '')))
                        p_loc_esc = html.escape(str(primary_ev.get('location', 'Page 1')))
                        st.markdown("##### 📌 Primary Evidence Candidate:")
                        st.markdown(f"""
                        <div class="evidence-box">
                            <p style="margin: 0; font-style: italic; color: #f8fafc;">
                                "{p_text_esc}"
                            </p>
                            <div style="margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center;">
                                <span class="source-meta-tag">📄 {p_src_esc}</span>
                                {get_auth_badge_html(p_auth)}
                                <span class="source-meta-tag">📍 {p_loc_esc}</span>
                                <span class="source-meta-tag">🎯 Similarity: {primary_ev.get('similarity', 0.0):.2f}</span>
                                <span class="source-meta-tag">⚡ Ranking: {primary_ev.get('ranking_score', primary_ev.get('similarity', 0.0)):.2f}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    if conflicting_evs:
                        st.markdown("##### 🔴 Contradicting Evidence Passages:")
                        for c_ev in conflicting_evs:
                            c_auth = c_ev.get("authority_level", DEFAULT_AUTHORITY_LEVEL)
                            c_text_esc = html.escape(str(c_ev.get('text', '')))
                            c_src_esc = html.escape(str(c_ev.get('source', '')))
                            c_loc_esc = html.escape(str(c_ev.get('location', 'Page 1')))
                            st.markdown(f"""
                            <div class="conflict-evidence-box">
                                <p style="margin: 0; font-style: italic; color: #fecaca;">
                                    "{c_text_esc}"
                                </p>
                                <div style="margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center;">
                                    <span class="source-meta-tag" style="border: 1px solid #7f1d1d;">📄 {c_src_esc}</span>
                                    {get_auth_badge_html(c_auth)}
                                    <span class="source-meta-tag">📍 {c_loc_esc}</span>
                                    <span class="source-meta-tag">🎯 Similarity: {c_ev.get('similarity', 0.0):.2f}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                else:
                    # Standard Evidence Span Details
                    st.markdown("##### 📌 Ground-Truth Evidence Grounding:")
                    evidence_list = selected_claim.get("evidence", [])
                    if evidence_list:
                        top_ev = evidence_list[0]
                        loc_type = top_ev.get('location_type') or "location"
                        loc_val = top_ev.get('location') or (f"Page {top_ev.get('page')}" if top_ev.get('page') else "N/A")
                        file_fmt = str(top_ev.get('file_type') or top_ev.get('source', '').split('.')[-1]).upper()
                        top_auth = top_ev.get('authority_level', src_auth)
                        top_text_esc = html.escape(str(top_ev.get('text', '')))
                        top_src_esc = html.escape(str(top_ev.get('source', '')))
                        top_loc_esc = html.escape(str(loc_val))

                        st.markdown(f"""
                        <div class="evidence-box">
                            <p style="margin: 0; font-style: italic; color: #f8fafc;">
                                "{top_text_esc}"
                            </p>
                            <div style="margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center;">
                                <span class="source-meta-tag">📄 Source: {top_src_esc}</span>
                                {get_auth_badge_html(top_auth)}
                                <span class="source-meta-tag">🏷️ Format: {file_fmt}</span>
                                <span class="source-meta-tag">📍 {loc_type.capitalize()}: {top_loc_esc}</span>
                                <span class="source-meta-tag">🎯 Similarity: {top_ev.get('similarity', 0.0):.2f}</span>
                                <span class="source-meta-tag">⚡ Ranking: {top_ev.get('ranking_score', top_ev.get('similarity', 0.0)):.2f}</span>
                                <span class="source-meta-tag">🆔 Chunk: {top_ev.get('chunk_id', 'N/A')}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)


                        if len(evidence_list) > 1:
                            with st.expander(f"View {len(evidence_list)-1} other candidate passages"):
                                for ev in evidence_list[1:]:
                                    ev_loc = ev.get('location') or (f"Page {ev.get('page')}" if ev.get('page') else "")
                                    st.markdown(f"- *\"{ev.get('text')}\"*")
                                    st.caption(f"Source: `{ev.get('source')}` ({ev_loc}) | Authority: `{ev.get('authority_level')}` | Similarity: {ev.get('similarity', 0.0):.2f} | Ranking: {ev.get('ranking_score', 0.0):.2f}")
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
            st.markdown(f"**Input Hash (SHA-256)**: `{cert.get('input_hash', 'N/A')}`")
            st.markdown(f"**Embedding Model**: `{cert.get('configuration', {}).get('embedding_model', 'BAAI/bge-small-en-v1.5')}`")
            st.markdown(f"**Entailment Engine**: `{cert.get('configuration', {}).get('entailment_provider', 'heuristic_fallback')}`")
            st.markdown(f"**Execution Latency**: `{cert.get('execution_time_seconds', summary.get('total_time_seconds', 0.0)):.3f}s` (Avg `{summary.get('avg_time_per_claim_ms', 0.0):.1f}ms`/claim)")

            # Source Document Hashes
            st.markdown("##### 🔐 Source Integrity Audit:")
            sources = cert.get("sources", [])
            if sources:
                source_table = [
                    {
                        "Filename": s["filename"],
                        "Authority Tier": s.get("authority_level", DEFAULT_AUTHORITY_LEVEL),
                        "Weight": f"{s.get('authority_weight', 0.94):.2f}",
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


# TAB 2: EVALUATION DASHBOARD & BENCHMARKS
with tab_eval:
    st.markdown("### 📊 VERDICT Evaluation & Benchmark Suite")
    st.markdown("""
    This evaluation suite measures the statistical accuracy, retrieval precision, and entailment performance 
    of the VERDICT verification engine against ground-truth datasets.
    """)

    b_col1, b_col2 = st.columns([3, 1])
    with b_col1:
        st.caption("Ground Truth Benchmark Dataset: `CRD_Compliance_Directive` (Multi-Section Regulatory Standard)")
    with b_col2:
        run_bench_clicked = st.button("▶️ Run Live Benchmark", type="primary", use_container_width=True)

    # Load benchmark data (cached or fresh)
    bench_data = None
    if run_bench_clicked:
        with st.spinner("Executing benchmark test suite..."):
            try:
                resp = requests.get(f"{BACKEND_URL}/evaluation?fresh=true", timeout=60)
                if resp.status_code == 200:
                    bench_data = resp.json()
            except Exception:
                pass
            if not bench_data:
                bench_data = run_benchmark(save_results=True)
            st.session_state.benchmark_results = bench_data
            st.toast("Benchmark evaluation completed!", icon="📊")
    elif st.session_state.benchmark_results:
        bench_data = st.session_state.benchmark_results
    else:
        # Load cached benchmark results
        try:
            resp = requests.get(f"{BACKEND_URL}/evaluation?fresh=false", timeout=5)
            if resp.status_code == 200:
                bench_data = resp.json()
        except Exception:
            pass
        if not bench_data:
            bench_data = get_or_run_benchmark(fresh=False)
        st.session_state.benchmark_results = bench_data

    if bench_data:
        # Top KPI Metrics Cards
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        with kpi1:
            st.metric("Overall Accuracy", f"{bench_data.get('accuracy', 0.0):.1f}%", f"{bench_data.get('correct_verdicts', 0)}/{bench_data.get('total_test_cases', 0)} correct")
        with kpi2:
            st.metric("Retrieval Precision (Top-K)", f"{bench_data.get('retrieval_precision', 0.0):.1f}%", "Evidence hit rate")
        with kpi3:
            st.metric("Macro F1-Score", f"{bench_data.get('macro_f1', 0.0):.1f}%", "Unweighted mean")
        with kpi4:
            st.metric("Suite Latency", f"{bench_data.get('total_time_seconds', 0.0):.2f}s", f"{bench_data.get('total_test_cases', 0)} claims")
        with kpi5:
            st.metric("Avg Claim Latency", f"{bench_data.get('avg_time_per_claim_ms', 0.0):.1f}ms", "per claim")

        st.markdown("---")

        # Per-Class Metrics Table
        st.markdown("#### 📈 Per-Class Classification Performance")
        class_metrics = bench_data.get("class_metrics", {})
        if class_metrics:
            metrics_table = []
            for cls_name, m in class_metrics.items():
                metrics_table.append({
                    "Class": cls_name,
                    "Precision": f"{m.get('precision', 0.0):.1f}%",
                    "Recall": f"{m.get('recall', 0.0):.1f}%",
                    "F1-Score": f"{m.get('f1_score', 0.0):.1f}%",
                    "Support": m.get("support", 0),
                    "True Positives (TP)": m.get("true_positives", 0),
                    "False Positives (FP)": m.get("false_positives", 0),
                    "False Negatives (FN)": m.get("false_negatives", 0)
                })
            st.table(metrics_table)

        # Confusion Matrix Heatmap / Table
        st.markdown("#### 🎯 Confusion Matrix")
        st.caption("Rows: Actual Ground-Truth Labels | Columns: VERDICT Predicted Labels")
        cm = bench_data.get("confusion_matrix", {})
        if cm:
            cm_rows = []
            for true_cls in ["SUPPORTED", "REFUTED", "UNVERIFIED"]:
                cm_rows.append({
                    "Ground Truth": f"Actual {true_cls}",
                    "Pred SUPPORTED": cm.get(true_cls, {}).get("SUPPORTED", 0),
                    "Pred REFUTED": cm.get(true_cls, {}).get("REFUTED", 0),
                    "Pred UNVERIFIED": cm.get(true_cls, {}).get("UNVERIFIED", 0)
                })
            st.table(cm_rows)

        # Test Cases Detailed Inspection
        st.markdown("#### 🔬 Detailed Test Case Predictions")
        with st.expander("Inspect Individual Benchmark Test Cases & Outcomes", expanded=False):
            tc_list = bench_data.get("test_cases", [])
            for tc in tc_list:
                sym = "🟢 ✓" if tc.get("is_correct") else "🔴 ✗"
                st.markdown(f"**{sym} `{tc.get('claim_id')}`** — Ground Truth: **`{tc.get('ground_truth')}`** | Predicted: **`{tc.get('predicted')}`** ({tc.get('processing_time_ms', 0.0)}ms)")
                st.caption(f"Claim: *\"{tc.get('claim')}\"*")
                if tc.get("evidence_text"):
                    st.caption(f"Retrieved: *\"{tc.get('evidence_text')[:120]}...\"*")
                st.markdown("<hr style='margin: 6px 0; border: none; border-top: 1px dashed #334155;'>", unsafe_allow_html=True)

st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("VERDICT — Claim Verification & Evidence Analysis")
