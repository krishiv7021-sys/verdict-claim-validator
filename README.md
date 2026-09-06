# VERDICT — Claim Verification & Evidence Analysis

> VERDICT is an evidence-grounded verification system that checks AI-generated claims against source documents and provides precise, inspectable evidence for verification decisions.

---

## 1. Overview

**VERDICT** is an evidence-grounded claim verification and analysis engine designed to eliminate hallucination in AI-generated answers, executive briefings, and compliance summaries. Instead of providing broad document citations or hyperlinks that require manual browsing, VERDICT analyzes input text at the atomic sentence level, retrieves specific evidentiary passages from reference documents, determines whether each claim is **SUPPORTED**, **REFUTED**, or **UNVERIFIED**, and anchors every decision to cryptographic SHA-256 source hashes and structured verification certificates.

---

## 2. Problem

Large language models (LLMs) frequently generate articulate responses containing subtle factual errors, invented figures, transposed dates, or reversed policies. In production domains such as legal compliance, financial auditing, healthcare operations, and corporate governance:
- **A link is not proof**: Standard retrieval-augmented generation (RAG) systems output document URLs or file citations, forcing users to manually read entire files to verify factual accuracy.
- **Authority Disparities**: Enterprise repositories often contain conflicting materials spanning multiple authority levels—such as formal statutes versus internal drafts or third-party web notes.
- **Unverifiable Black Boxes**: LLMs rarely provide machine-readable, reproducible proof demonstrating how their output connects to authorized source materials.

---

## 3. Solution

VERDICT introduces an explicit, deterministic verification layer between AI text generation and downstream decision-makers:
- **Atomic Claim Extraction**: Decomposes compound AI-generated text into granular, individually testable factual assertions.
- **Multi-Format Ingestion**: Ingests and SHA-256 hashes source documents across 9 common enterprise formats with format-specific location tracking (`Page`, `Paragraph`, `Slide`, `Row`, `JSON Path`, or `Element`).
- **Authority-Weighted Semantic Retrieval**: Encodes claims and sources using `BAAI/bge-small-en-v1.5` embeddings, applying an authority-weighted ranking score where semantic relevance strictly dominates while document authority breaks ties.
- **Cross-Source Conflict Detection**: Identifies material factual disagreements across multiple source files on the same claim and isolates primary evidence from contradictory passages.
- **Conservative Decision Logic**: Claims with incomplete or inconclusive evidence are classified as **`UNVERIFIED`** rather than failing upward into supported. Any refuted claim or detected conflict escalates the document to **`REVIEW_REQUIRED`**.
- **Cryptographic Auditability**: Issues tamper-evident JSON certificates and exportable Markdown audit reports.

---

## 4. Key Features

- **Format-Aware Evidence Localization**: Locates evidence spans down to specific structural units across 9 standard file formats.
- **Hierarchical Source Authority**: Configurable authority tiers (`STATUTORY`, `POLICY`, `INTERNAL`, `REFERENCE`) with transparent, non-overpowering ranking formulas.
- **Cross-Source Contradiction Isolation**: Detects multi-document factual conflicts (e.g., Source A states 7 days; Source B states 10 days) and tags conflicting evidence.
- **Conservative Classification**: Three strict verdicts (`SUPPORTED`, `REFUTED`, `UNVERIFIED`). Overall status evaluates to `VERIFIED` only if 100% of claims are supported without conflict.
- **Cryptographic Audit Trail**: SHA-256 checksums of all source files and input text, stored in persistent JSON certificates.
- **Performance & Latency Tracking**: Records per-claim verification latency in milliseconds and total pipeline execution time.
- **Interactive Evaluation Dashboard**: Built-in benchmark suite reporting Accuracy, Retrieval Precision, Macro F1, Per-Class F1 metrics, Confusion Matrix, and latency.
- **Dual Engine Architecture**: Local deterministic heuristic NLI engine (zero API keys required) with optional cloud LLM providers (Groq, OpenAI, Google Gemini).

---

## 5. Supported Document Formats

VERDICT features a format-aware ingestion pipeline supporting 9 common file formats:

| Format | File Extension | Extraction Library | Tracked Structural Location | Example Location Label |
| :--- | :--- | :--- | :--- | :--- |
| **PDF** | `.pdf` | `pypdf` | Page number | `Page 4` |
| **DOCX** | `.docx`, `.doc` | `python-docx` | Paragraph / Table index | `Paragraph 12`, `Table 1, Row 2` |
| **PPTX** | `.pptx`, `.ppt` | `python-pptx` | Slide number | `Slide 5` |
| **XLSX** | `.xlsx`, `.xls` | `openpyxl` | Sheet & Row / Cell index | `Revenue_Model!Row 14` |
| **CSV** | `.csv` | Standard library `csv` | Row index | `Row 8` |
| **Markdown** | `.md`, `.markdown` | Structural parser | Section heading | `Section 'Compliance Protocols'` |
| **JSON** | `.json` | Standard library `json` | Hierarchical JSON Path | `$.filing_rules.mandatory_deadline` |
| **HTML** | `.html`, `.htm` | `beautifulsoup4` | Semantic HTML element | `<article> #filing-policy`, `<p>` |
| **TXT** | `.txt` | Multi-encoding reader | Paragraph / Section | `Paragraph 2` |

---

## 6. How VERDICT Works

The verification process follows five sequential stages:

```
Source Files ──────► SHA-256 Hashing & Ingestion ──────► Chunking with Metadata
                                                               │
AI Draft Text ────► Atomic Claim Decomposition                ▼
                            │                   Embedding Generation (BGE-Small)
                            │                                  │
                            ▼                                  ▼
                   Vector Retrieval & Authority Ranking ◄───────
                            │
                            ▼
              Entailment & Conflict Analysis (NLI / LLM)
                            │
                            ▼
           Conservative Verdict Aggregation & Certification
```

1. **Document Ingestion & Cryptographic Checksumming**: Each uploaded reference file is hashed using SHA-256 and parsed into structured text chunks retaining format-specific location metadata.
2. **Atomic Claim Decomposition**: The input AI-generated text is segmented into atomic, testable claims using rule-based sentence boundary detection with abbreviation handling and compound sentence splitting.
3. **Authority-Weighted Dense Retrieval**: Claims and document chunks are converted into dense vector embeddings using `BAAI/bge-small-en-v1.5`. Top-$k$ candidate passages are ranked by combining cosine similarity with source authority weights.
4. **Entailment & Conflict Evaluation**: Candidate passages are verified against each claim:
   - **`SUPPORTED`**: The evidence directly confirms the claim's factual assertion.
   - **`REFUTED`**: The evidence directly contradicts key metrics, dates, obligations, or polarity.
   - **`UNVERIFIED`**: The evidence is ambiguous, partial, or absent.
   - If multiple sources provide contradictory evidence, a conflict flag is raised, and both primary and conflicting passages are isolated.
5. **Conservative Aggregation & Certification**: If any claim is `REFUTED` or `UNVERIFIED`, or if a cross-source conflict is detected, the overall document verdict is flagged as **`REVIEW_REQUIRED`**. Only if all claims are supported without conflict is the status set to **`VERIFIED`**.

---

## 7. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             USER INTERFACE                                  │
│             Streamlit Web Application (app.py) / REST API                   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FASTAPI BACKEND SERVICE                            │
│  • POST /verify     • GET /evaluation     • GET /certificate/{id}           │
│  • GET /health      • GET /demo           • GET /certificate/{id}/report    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VERIFICATION PIPELINE ENGINE                         │
│                                                                             │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐  │
│  │   Document Parser    │  │   Claim Decomposer   │  │  SHA-256 Hasher  │  │
│  │ (9 Formats + Location│  │ (Atomic Segmentation │  │ (Tamper-Evident  │  │
│  │     Tracking)        │  │   & Rule Splitting)  │  │   Audit Trail)   │  │
│  └──────────┬───────────┘  └──────────┬───────────┘  └────────┬─────────┘  │
│             │                         │                       │             │
│             ▼                         ▼                       │             │
│  ┌────────────────────────────────────────────────┐           │             │
│  │               Embedding Retriever              │           │             │
│  │   • BAAI/bge-small-en-v1.5 Vector Search       │           │             │
│  │   • Authority-Weighted Ranking Formula         │           │             │
│  └────────────────────────┬───────────────────────┘           │             │
│                           │                                   │             │
│                           ▼                                   │             │
│  ┌────────────────────────────────────────────────┐           │             │
│  │            NLI & Entailment Engine             │           │             │
│  │   • Deterministic Heuristic NLI Fallback       │           │             │
│  │   • Optional LLM (Groq / OpenAI / Gemini)      │           │             │
│  │   • Cross-Source Conflict Detection            │           │             │
│  └────────────────────────┬───────────────────────┘           │             │
│                           │                                   │             │
│                           ▼                                   │             │
│  ┌────────────────────────────────────────────────┐           │             │
│  │        Conservative Verdict Aggregator         │           │             │
│  │    (VERIFIED vs. REVIEW_REQUIRED Escalation)   │           │             │
│  └────────────────────────┬───────────────────────┘           │             │
│                           │                                   │             │
│                           ▼                                   ▼             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     Certificate Synthesis Engine                      │  │
│  │   • Verification Certificate JSON (v1.1)                              │  │
│  │   • Human-Readable Markdown Audit Report                              │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Technology Stack

- **Core Backend**: Python 3.9+, FastAPI, Uvicorn, Pydantic v2
- **Frontend Dashboard**: Streamlit, Custom Responsive CSS
- **Vector Embeddings**: `BAAI/bge-small-en-v1.5` via `sentence-transformers` (with deterministic subword TF-IDF fallback)
- **Document Extractors**:
  - PDF: `pypdf`
  - DOCX: `python-docx`
  - PPTX: `python-pptx`
  - XLSX: `openpyxl`
  - HTML: `beautifulsoup4`
  - CSV, TXT, JSON, Markdown: Python standard libraries and structural regex
- **Entailment Analysis**:
  - Deterministic Rule-Based NLI: Handles numeric contradiction detection, obligation/prohibition inversion, negation checking, and stemmed keyword overlap.
  - Cloud Providers (Optional): Groq (`llama-3.3-70b-versatile`), OpenAI (`gpt-4o-mini`), Google Gemini (`gemini-1.5-flash`).
- **Testing & Quality Assurance**: Pytest (36 automated tests), benchmark evaluation suite.

---

## 9. Verification Workflow

1. **Provide AI Draft**: Paste AI-generated text or upload a `.txt` draft into the left input panel.
2. **Attach Source Documents**: Upload one or more reference files (`.pdf`, `.docx`, `.pptx`, `.xlsx`, `.csv`, `.txt`, `.md`, `.json`, `.html`).
3. **Configure Authority Tiers (Optional)**: Assign an authority level to each document (`STATUTORY`, `POLICY`, `INTERNAL`, `REFERENCE`).
4. **Execute Verification**: Click **VERIFY CLAIMS**. The system executes ingestion, hashing, decomposition, retrieval, conflict detection, and entailment.
5. **Inspect Claims**:
   - Check executive summary metrics and overall verdict status (`VERIFIED` vs `REVIEW REQUIRED`).
   - Click any individual claim to inspect its primary evidence span, source authority badge, similarity score, ranking score, and entailment rationale.
   - If a cross-source conflict is detected, review the highlighted contradictory passages.
6. **Export Verification Artifacts**: Download the JSON Verification Certificate or Markdown Audit Report.

---

## 10. Source Authority

VERDICT models document reliability through 4 extensible authority tiers:

| Authority Tier | Weight | Category Description | Typical Document Types |
| :--- | :---: | :--- | :--- |
| **`STATUTORY / OFFICIAL`** | `1.00` | Binding legislation, formal statutes, legal decrees | Legislative acts, primary contracts, court orders |
| **`POLICY / REGULATION`** | `0.97` | Organizational governance standards, compliance policies | Regulatory directives, compliance manuals, bylaws |
| **`INTERNAL / ORGANIZATIONAL`** | `0.94` | Standard operating procedures, internal memos *(Default)* | Employee handbooks, runbooks, internal wiki notes |
| **`REFERENCE / WEB`** | `0.90` | Third-party analysis, general web references, documentation | Industry guides, external whitepapers, articles |

### Transparent, Non-Overpowering Ranking Formula
To prevent high-authority documents from matching unrelated claims, semantic relevance remains dominant (minimum 75% baseline weight):

$$\text{Ranking Score} = \text{round}(\text{Cosine Similarity} \times (0.75 + 0.25 \times \text{Authority Weight}), 4)$$

- A statutory document ($\text{wt} = 1.00$) with similarity $0.60$ achieves a ranking score of $0.6000$.
- A web reference ($\text{wt} = 0.90$) with similarity $0.85$ achieves a ranking score of $0.8288$.
- Relevant evidence consistently outranks irrelevant text regardless of authority, while higher-authority sources win between closely competing relevant passages.

---

## 11. Conflict Detection

When multiple reference files contain differing information regarding the same claim, VERDICT's multi-candidate evaluation identifies the contradiction:
1. **Contradiction Detection**: If a candidate passage from Source A supports the claim while another candidate from Source B refutes the claim (e.g., conflicting notice periods or differing penalty amounts), the engine flags `conflict_detected: True`.
2. **Evidence Separation**: The system separates results into:
   - `primary_evidence`: The highest-ranked candidate passage grounding the claim.
   - `supporting_evidence`: Corroborating passages from other documents.
   - `conflicting_evidence`: Contradictory passages with source names and authority levels.
3. **Escalation**: Any claim with a detected conflict automatically flags the document status as **`REVIEW_REQUIRED`** and includes conflict audit notes in the generated certificate.

---

## 12. Verification Certificates

Every verification run produces a machine-readable JSON Certificate (specification v1.1) containing complete cryptographic provenance:

```json
{
  "certificate_version": "1.1",
  "certificate_id": "vc_29d8b4625da7",
  "timestamp": "2025-01-15T10:30:00Z",
  "input_hash": "a680cd1822a1383c75c51ac5ce3ef0536acbc920c57fd272a48fdad09ba0cd43",
  "execution_time_seconds": 2.095,
  "overall_verdict": "REVIEW_REQUIRED",
  "summary": {
    "total_claims": 2,
    "supported": 2,
    "refuted": 0,
    "unverified": 0,
    "conflicts_detected": 1,
    "total_sources": 3,
    "total_time_seconds": 2.095,
    "avg_time_per_claim_ms": 1047.7
  },
  "configuration": {
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "entailment_provider": "heuristic_fallback",
    "top_k": 3,
    "similarity_threshold": 0.25,
    "authority_weighting_enabled": true,
    "conflict_detection_enabled": true
  },
  "sources": [
    {
      "source_id": "SRC_a982f1bc",
      "filename": "Statutory_Act.txt",
      "sha256": "4a72d3f9e8a01b2c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c",
      "file_type": "txt",
      "authority_level": "STATUTORY / OFFICIAL",
      "authority_weight": 1.00,
      "page_count": 1,
      "evidence_locations": ["Paragraph 1"]
    }
  ],
  "claims": [
    {
      "claim_id": "C001",
      "claim_text": "The statutory termination notice period is 7 days.",
      "verdict": "SUPPORTED",
      "confidence": 0.92,
      "explanation": "Source conflict detected: Grounded by Statutory_Act.txt, but contradicted by Industry_Guide.txt.",
      "source_authority": "STATUTORY / OFFICIAL",
      "similarity_score": 0.882,
      "ranking_score": 0.882,
      "conflict_detected": true,
      "processing_time_ms": 0.4,
      "primary_evidence": {
        "source": "Statutory_Act.txt",
        "file_type": "txt",
        "location": "Paragraph 1",
        "text": "The statutory termination notice period is strictly 7 days for all standard contracts.",
        "similarity": 0.882,
        "ranking_score": 0.882,
        "authority_level": "STATUTORY / OFFICIAL",
        "authority_weight": 1.00
      },
      "conflicting_evidence": [
        {
          "source": "Industry_Guide.txt",
          "file_type": "txt",
          "location": "Paragraph 1",
          "text": "Standard notice periods in industry generally range around 30 days.",
          "similarity": 0.717,
          "authority_level": "REFERENCE / WEB"
        }
      ]
    }
  ]
}
```

---

## 13. Evaluation

VERDICT includes an automated benchmark evaluation suite (`evaluation/evaluate.py`) that benchmarks verification accuracy and retrieval precision against ground-truth datasets.

### Benchmark Results Summary

- **Overall Verification Accuracy**: `70.0%` (7 / 10 test cases correct)
- **Evidence Retrieval Precision (Top-K)**: `100.0%` (8 / 8 keyword-targeted cases matched)
- **Macro F1-Score**: `65.3%`
- **Average Claim Latency**: `~400 ms` per claim

### Per-Class Performance Breakdown

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **`SUPPORTED`** | 80.0% | 100.0% | 88.9% | 4 |
| **`REFUTED`** | 66.7% | 50.0% | 57.1% | 4 |
| **`UNVERIFIED`** | 50.0% | 50.0% | 50.0% | 2 |

### Confusion Matrix

| Ground Truth \ Predicted | Pred SUPPORTED | Pred REFUTED | Pred UNVERIFIED |
| :--- | :---: | :---: | :---: |
| **Actual SUPPORTED** | 4 | 0 | 0 |
| **Actual REFUTED** | 1 | 2 | 1 |
| **Actual UNVERIFIED** | 0 | 1 | 1 |

*Note: These metrics reflect deterministic rule-based NLI inference without external LLM API dependencies.*

---

## 14. Installation

### Prerequisites
- Python 3.9+ (Python 3.9–3.11 recommended)
- `pip` package manager

### Setup Steps
```bash
# Clone the repository
git clone https://github.com/your-username/verdict-claim-verification.git
cd verdict-claim-verification

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### Environment Configuration (Optional)
To use cloud LLM providers for entailment verification instead of the built-in deterministic NLI fallback, copy `.env.example` to `.env` and provide your API keys:
```bash
cp .env.example .env
```
*(If no API keys are configured, VERDICT automatically falls back to its built-in local deterministic NLI engine, requiring no external credentials).*

---

## 15. Running Locally

### 1. Launch FastAPI Backend
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be live at `http://localhost:8000/docs`.

### 2. Launch Streamlit Frontend
```bash
streamlit run frontend/app.py
```
Open your browser at `http://localhost:8501`.

### 3. Run Pipeline Demonstrations
```bash
# Multi-format verification across all 9 document types
python scripts/verify_multi_format_pipeline.py

# Advanced verification demonstration (authority weighting + cross-source conflicts)
python scripts/demo_advanced_verification.py
```

---

## 16. API Usage

VERDICT exposes a RESTful API via FastAPI:

| Method | Endpoint | Description | Parameters |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | API status, metadata, and available endpoints | None |
| `GET` | `/health` | Service health, active embedding model, entailment provider | None |
| `GET` | `/demo` | Bundled demo draft and source document for testing | None |
| `POST` | `/verify` | Execute verification on draft text and source files | `draft_text`, `draft_file`, `source_files`, `source_authorities` |
| `GET` | `/evaluation` | Benchmark evaluation metrics, F1 scores, confusion matrix | `fresh` (optional boolean, default `false`) |
| `GET` | `/certificate/{id}` | Retrieve stored JSON verification certificate | `id` (path) |
| `GET` | `/certificate/{id}/report` | Retrieve human-readable Markdown report | `id` (path) |
| `POST` | `/certificate/{id}/export` | Export downloadable JSON certificate data | `id` (path) |

### Verification Example (`curl`)

```bash
curl -X POST "http://localhost:8000/verify" \
  -F "draft_text=Annual compliance filings are due within 30 days." \
  -F "source_files=@compliance_directive.pdf" \
  -F 'source_authorities={"compliance_directive.pdf": "STATUTORY / OFFICIAL"}'
```

---

## 17. Project Structure

```
.
├── backend/
│   ├── main.py                  # FastAPI server and REST endpoints
│   ├── schemas.py               # Pydantic v2 data models and enums
│   ├── services/
│   │   ├── certificate.py       # Certificate synthesis and Markdown report generator
│   │   ├── claim_decomposer.py  # Rule-based atomic claim segmentation
│   │   ├── document_parser.py   # Ingestion layer for 9 document formats
│   │   ├── embeddings.py        # Vector embedding wrapper (BGE-Small / TF-IDF)
│   │   ├── entailment.py        # Deterministic NLI & LLM entailment engines
│   │   ├── hashing.py           # Cryptographic SHA-256 computation
│   │   ├── retriever.py         # Authority-weighted candidate retrieval
│   │   └── verifier.py          # Master verification orchestration pipeline
│   └── utils/
│       └── helpers.py           # Demo text and shared utilities
├── data/
│   ├── demo_files/              # Multi-format demo files (DOCX, PPTX, XLSX, CSV, etc.)
│   ├── sample_policy.pdf        # Ground-truth PDF reference document
│   └── sample_policy.txt        # Ground-truth TXT reference document
├── evaluation/
│   ├── benchmark_data.json      # Ground-truth evaluation dataset (10 test cases)
│   ├── benchmark_results.json   # Cached evaluation metrics and confusion matrix
│   └── evaluate.py              # Benchmark execution script and metrics calculator
├── frontend/
│   └── app.py                   # Streamlit verification interface and evaluation dashboard
├── scripts/
│   ├── demo_advanced_verification.py # Authority weighting and conflict demonstration
│   ├── generate_all_demo_formats.py  # Generator for multi-format demo files
│   ├── generate_sample_files.py      # Generator for sample PDF/TXT files
│   ├── verify_demo_pipeline.py       # End-to-end verification verification script
│   └── verify_multi_format_pipeline.py # 9-format pipeline test script
├── tests/
│   ├── conftest.py              # Pytest configuration and shared fixtures
│   ├── test_advanced_features.py# Tests for authority weighting and conflict detection
│   ├── test_api.py              # FastAPI endpoint tests
│   ├── test_claim_decomposer.py # Sentence splitting and claim extraction tests
│   ├── test_document_parser.py  # TXT and PDF parser unit tests
│   ├── test_hashing.py          # SHA-256 integrity and tamper detection tests
│   ├── test_new_formats.py      # DOCX, PPTX, XLSX, CSV, MD, JSON, HTML parser tests
│   └── test_pipeline.py         # End-to-end pipeline and verdict logic tests
├── .env.example                 # Template for environment configuration
├── .gitignore                   # Git exclusion rules for secrets, caches, and reports
├── requirements.txt             # Pinned project dependencies
└── README.md                    # Project documentation
```

---

## 18. Limitations

- **Scanned Document OCR**: Pure raster image files (PNG, JPG, TIFF) and non-text scanned PDFs currently return a structured notice rather than attempting ungrounded character recognition.
- **Encrypted Archives**: Password-protected PDF and Office documents must be decrypted before ingestion.
- **Complex Tabular Relational Joins**: High-density spreadsheets and nested table structures are parsed sequentially by rows and cells; relational multi-table cross-joins require explicit document schema definitions.

---

## 19. Future Improvements

- **Native OCR Engine**: Integration with open-source OCR libraries (e.g., Tesseract or PaddleOCR) for scanned PDFs and embedded document images.
- **Citation Insertion**: Automatically injecting grounded inline footnote citations back into the original draft text.
- **Graph-Based Multi-Hop Reasoning**: Linking multi-document dependency chains where Claim A relies on Document X and Document Y simultaneously.
- **Batch Repository Verification**: Command-line interface for running bulk verification suites over enterprise document repositories.

---

## Testing & Quality Assurance

Run the complete automated test suite (36 tests across unit, integration, and API layers):
```bash
pytest -v tests/
```

Run the benchmark evaluation runner:
```bash
python evaluation/evaluate.py
```
