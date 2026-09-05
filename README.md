# VERDICT — Claim Verification & Evidence Analysis

> **"A link is not proof."**  
> An evidence-grounded verification system that checks AI-generated claims against source documents, resolves multi-tier authority hierarchies, detects cross-source factual conflicts, and provides precise, inspectable evidence for every verification decision.

---

## 📌 1. What VERDICT Does

Large language models frequently generate articulate answers with subtle, factual hallucinations. Typical retrieval-augmented systems and search interfaces only return a list of links or document citations, forcing users to manually inspect entire files to verify whether a claim is actually substantiated. Furthermore, real-world enterprise repositories often contain contradictory guidelines, draft documents, or varying authority tiers (such as statutory regulations vs. internal memos).

**VERDICT** transforms verification into a first-class, verifiable, and cryptographically anchored process:
- **Decomposes AI-generated text** into granular atomic claims.
- **Ingests and SHA-256 hashes** reference source documents across 9 standard formats.
- **Hierarchical Source Authority Weighting**: Distinguishes statutory laws (1.00), regulatory policies (0.97), internal procedures (0.94), and web references (0.90) so authoritative sources take precedence without overpowering semantic relevance.
- **Cross-Source Conflict Detection**: Detects material contradictions across multiple source documents on the same claim (e.g., Source A specifies 7 days; Source B specifies 10 days) and isolates primary evidence vs conflicting evidence.
- **Dense Vector Retrieval**: Performs sentence- and chunk-level retrieval using `BAAI/bge-small-en-v1.5`.
- **Strict Entailment Verification**: Applies conservative decision logic (`SUPPORTED`, `REFUTED`, `UNVERIFIED`).
- **Format-Aware Evidence Location**: Pins each claim to its exact evidence location (`Page`, `Paragraph`, `Slide`, `Row`, `Cell`, `JSON Path`, or `Element`).
- **Cryptographic Audit Trail**: Computes input hashes, source checksums, per-claim latency metrics, and generates machine-readable Verification Certificates and Markdown audit reports.
- **Interactive Evaluation Dashboard**: Evaluates model precision, recall, F1-scores per class, confusion matrices, and latency on ground-truth benchmark datasets.

---

## 🎯 2. Why Claim Verification is Important

In compliance, legal, financial, healthcare, and enterprise research workflows, an AI system that hallucinates numbers, dates, terms, or policies creates significant legal and regulatory risk.

1. **Hallucination Mitigation**: AI models state contradictory facts with high grammatical confidence. VERDICT isolates each claim and checks for explicit evidence grounding.
2. **Authority & Conflict Awareness**: Different documents carry different legal weights. Outdated draft notes or low-authority memos should not override statutory directives. When sources disagree, VERDICT highlights both sides and flags the conflict for human review.
3. **Auditability & Compliance**: Enterprises cannot accept untraceable "black-box" summaries. They require verifiable audit trails linking claims to specific lines in authorized reference materials.
4. **Reproducibility**: By computing SHA-256 digests of all ingested sources and generating structured verification certificates, decisions can be independently verified and audited at any later time.

---

## 🏛️ 3. Source Authority Tiers & Ranking Formula

VERDICT categorizes source documents into 4 extensible authority tiers:

| Authority Tier | Weight | Description | Typical Use Cases |
| :--- | :---: | :--- | :--- |
| **`STATUTORY / OFFICIAL`** | `1.00` | Binding legislation, court rulings, statutory directives | Primary compliance laws, formal contracts |
| **`POLICY / REGULATION`** | `0.97` | Corporate standards, regulator guidelines, bylaws | Compliance manuals, ISO standards |
| **`INTERNAL / ORGANIZATIONAL`** | `0.94` | Standard operating procedures, memos, runbooks *(Default)* | Employee handbooks, internal wikis |
| **`REFERENCE / WEB`** | `0.90` | Third-party analysis, market guides, general web references | Trade articles, general research |

### Transparent, Non-Overpowering Ranking Formula
To ensure that an authoritative document cannot be falsely matched to an unrelated claim, semantic relevance remains dominant (minimum 75% baseline weight):
$$\text{Ranking Score} = \text{Cosine Similarity} \times (0.75 + 0.25 \times \text{Authority Weight})$$

- A statutory document ($\text{wt} = 1.00$) with similarity $0.60$ achieves a ranking score of $0.6000$.
- A web reference ($\text{wt} = 0.90$) with similarity $0.85$ achieves a ranking score of $0.8288$.
- Therefore, a highly relevant reference source will correctly outrank an irrelevant statutory clause, while between two similarly relevant passages, the higher-authority source wins.

---

## ⚔️ 4. Cross-Source Conflict Detection

When multiple source documents discuss the same claim, VERDICT automatically analyzes candidate passages across different files:
1. **Contradiction Analysis**: If Passage 1 from Source A supports the claim, but Passage 2 from Source B contradicts the claim (e.g. numeric differences, prohibition vs permission), VERDICT flags `conflict_detected: True`.
2. **Evidence Separation**: Separates evidence into `primary_evidence`, `supporting_evidence`, and `conflicting_evidence`.
3. **Conservative Escalation**: Any detected conflict automatically escalates the overall document verdict to **`REVIEW_REQUIRED`** with an audit warning:
   > `⚠️ Conflicting Evidence Detected Across Sources: Grounded by Source A, but materially contradicted by Source B. Review required.`

---

## ⚙️ 5. How the System Works

The VERDICT verification engine operates through five sequential stages:

1. **Document Ingestion & Cryptographic Hashing**:
   - Computes a canonical SHA-256 hash of each uploaded reference document.
   - Extracts clean text while preserving format-specific structural locations (`Page 4`, `Paragraph 12`, `Slide 3`, `Revenue!Row 5`, etc.).
2. **Claim Decomposition**:
   - Parses the input AI draft text into self-contained atomic claims using rule-based sentence boundary detection with abbreviation awareness and compound sentence splitting.
   - Assigns unique identifiers (`C001`, `C002`, ...).
3. **Evidence Retrieval**:
   - Encodes source chunks and claims into normalized dense vectors using `BAAI/bge-small-en-v1.5`.
   - Performs cosine similarity search to retrieve the top-$k$ most relevant candidate passages per claim.
4. **Entailment Analysis & Verdict Assignment**:
   - Evaluates retrieved candidates against the claim using an LLM provider (Groq, OpenAI, Gemini) or a high-precision deterministic NLI engine.
   - Applies strict three-verdict classification:
     - **`SUPPORTED`**: Source text directly confirms the factual assertion of the claim.
     - **`REFUTED`**: Source text directly contradicts numeric metrics, dates, obligations, or polarity.
     - **`UNVERIFIED`**: Source evidence is ambiguous, partial, or absent. **Never fails upward into supported.**
5. **Conservative Aggregation & Certification**:
   - Evaluates overall document status: if **any** claim is `REFUTED` or `UNVERIFIED`, the document is flagged as **`REVIEW_REQUIRED`**. Only if **all** claims are `SUPPORTED` does it receive **`VERIFIED`**.
   - Issues a JSON Verification Certificate with complete metadata and exports a Markdown audit report.

---

## 📁 4. Supported Document Formats

VERDICT features a format-aware ingestion layer supporting 9 common file formats:

| Format | File Extension | Extractor Library | Tracked Location | Example Location |
| :--- | :--- | :--- | :--- | :--- |
| **PDF** | `.pdf` | `pypdf` | Page number | `Page 4` |
| **DOCX** | `.docx`, `.doc` | `python-docx` | Paragraph / Table index | `Paragraph 32`, `Table 1, Row 2` |
| **PPTX** | `.pptx`, `.ppt` | `python-pptx` | Slide number | `Slide 7` |
| **XLSX** | `.xlsx`, `.xls` | `openpyxl` | Sheet & Row / Cell | `Revenue_Model!Row 14` |
| **CSV** | `.csv` | `csv` (Standard Lib) | Row index | `Row 18` |
| **Markdown** | `.md`, `.markdown` | Structural RegEx | Heading section | `Section 'Filing Protocols'` |
| **JSON** | `.json` | `json` (Standard Lib) | Hierarchical JSON Path | `$.company.filing_rules.deadline` |
| **HTML** | `.html`, `.htm` | `beautifulsoup4` | Semantic HTML element | `<article> #filing-guidance`, `<p>` |
| **TXT** | `.txt` | Multi-encoding reader | Paragraph / Section | `Paragraph 1`, `Section 2` |

---

## 🏗️ 5. Architecture

```
                               ┌────────────────────────┐
                               │  AI-Generated Draft    │
                               │  (Text / Uploaded TXT) │
                               └───────────┬────────────┘
                                           │
                                           ▼
┌───────────────────────┐       ┌───────────────────────┐
│ Multi-Format Sources  │       │   Claim Decomposer    │
│(PDF, DOCX, PPTX, XLSX,│       │ (Sentence Splitting & │
│ CSV, TXT, MD, JSON,   │       │  Atomic C001.. C00n)  │
│ HTML)                 │       └───────────┬───────────┘
└──────────┬────────────┘                   │
     ┌─────┴──────────────┐                 │
     ▼                    ▼                 │
┌───────────────┐ ┌───────────────┐         │
│ SHA-256 Hasher│ │Document Parser│         │
│ (Audit Trail) │ │(Page & Chunks)│         │
└───────┬───────┘ └───────┬───────┘         │
        │                 │                 │
        │                 ▼                 │
        │       ┌───────────────────┐       │
        │       │  Embedding Model  │       │
        │       │bge-small-en-v1.5  │       │
        │       └─────────┬─────────┘       │
        │                 │                 │
        │                 ▼                 ▼
        │       ┌───────────────────────────────────┐
        │       │ Vector Similarity Retriever       │
        │       │ (Top-K Evidence Candidates)       │
        │       └─────────────────┬─────────────────┘
        │                         │
        │                         ▼
        │       ┌───────────────────────────────────┐
        │       │      Strict Entailment Engine     │
        │       │   (Groq / OpenAI / Gemini / NLI)  │
        │       └─────────────────┬─────────────────┘
        │                         │
        │                         ▼
        │       ┌───────────────────────────────────┐
        │       │   Conservative Verdict Aggregator │
        │       │ (VERIFIED vs. REVIEW_REQUIRED)    │
        │       └─────────────────┬─────────────────┘
        │                         │
        └────────────────► ◄──────┘
                           │
                           ▼
        ┌───────────────────────────────────────────┐
        │       Reproducible Verification           │
        │              Certificate                  │
        │  • Machine-Readable JSON Certificate      │
        │  • Human-Readable Audit Report (Markdown) │
        │  • Interactive Streamlit Review UI        │
        └───────────────────────────────────────────┘
```

---

## 💻 6. Technologies Used

- **Backend**: Python 3.9+, FastAPI, Uvicorn, Pydantic v2
- **Frontend**: Streamlit, Custom Responsive CSS
- **Embeddings**: `BAAI/bge-small-en-v1.5` via `sentence-transformers` (with deterministic subword TF-IDF fallback)
- **Document Extractors**: `pypdf`, `python-docx`, `python-pptx`, `openpyxl`, `beautifulsoup4`
- **Entailment Engine**:
  - LLM Providers: Groq (`llama-3.3-70b-versatile`), OpenAI (`gpt-4o-mini`), Google Gemini (`gemini-2.5-flash`)
  - Local NLI Engine: Deterministic rule-based engine handling numeric discrepancy detection, permission/prohibition inversion, negation flips, and stemmed keyword overlap
- **Testing & Benchmarking**: Pytest (29 tests), custom benchmark suite

---

## 🚀 7. How to Run It Locally

### Prerequisites
- Python 3.9+ (Python 3.9–3.11 recommended)
- `pip` and virtual environment support

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/verdict-claim-verification.git
cd verdict-claim-verification

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration (Optional)
To use cloud LLM providers for entailment verification, copy `.env.example` to `.env` and provide your API keys:
```bash
cp .env.example .env
```
*(If no API keys are configured, VERDICT automatically falls back to its built-in local deterministic NLI engine, requiring no external credentials).*

### Launching the Application

**Terminal 1 — FastAPI Backend**:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be live at `http://localhost:8000/docs`.

**Terminal 2 — Streamlit Frontend**:
```bash
streamlit run frontend/app.py
```
Open your browser at `http://localhost:8501`.

---

## 📋 8. Verification Workflow

1. **Load AI Draft**: Paste AI-generated text or upload a `.txt` draft into the left input panel.
2. **Attach Source Documents**: Upload one or more reference files (`.pdf`, `.docx`, `.pptx`, `.xlsx`, `.csv`, `.txt`, `.md`, `.json`, `.html`).
3. **Execute Verification**: Click **VERIFY CLAIMS**. The system displays a live 5-stage progress indicator:
   - Ingesting & Hashing Source Documents
   - Decomposing Draft into Atomic Claims
   - Generating Semantic Dense Vectors
   - Retrieving Evidence Passages
   - Executing Entailment & Synthesizing Certificate
4. **Inspect Claims**:
   - Review executive summary metrics and overall verdict badge (`VERIFIED` vs `REVIEW REQUIRED`).
   - Click on any claim in the list to open the **Evidence Inspector**, showing exact ground-truth passage, format badge, source filename, location (`Page`, `Paragraph`, `Slide`, `Row`, etc.), cosine similarity, and entailment rationale.
5. **Download Verification Artifacts**:
   - Download the tamper-evident JSON Certificate.
   - Download the human-readable Markdown Audit Report.

---

## 📜 9. Verification Certificate

Every completed verification generates a cryptographic certificate (v1.1) containing full provenance, authority tiers, conflict audit, and timing metrics:

```json
{
  "certificate_version": "1.1",
  "certificate_id": "vc_29d8b4625da7",
  "timestamp": "2026-09-05T12:00:00Z",
  "input_hash": "a680cd1822a1383c75c51ac5ce3ef0536acbc920c57fd272a48fdad09ba0cd43",
  "execution_time_seconds": 2.095,
  "overall_verdict": "REVIEW_REQUIRED",
  "summary": {
    "total_claims": 2,
    "supported": 2,
    "refuted": 0,
    "unverified": 0,
    "conflicts_detected": 2,
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
      "sha256": "4a72d3f9e8a01b2c...",
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
      "explanation": "Source conflict detected: Grounded by Statutory_Act.txt (Paragraph 1, STATUTORY / OFFICIAL), but materially contradicted by Industry_Guide.txt (Paragraph 1, REFERENCE / WEB). Review required.",
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

## 🌐 10. REST API Endpoints

VERDICT provides a high-performance FastAPI backend with automatic OpenAPI documentation:

| Method | Endpoint | Description | Key Parameters |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | System health, active embedding model & entailment provider | None |
| `GET` | `/demo` | Bundled demo draft and source document for instant demo | None |
| `POST` | `/verify` | Execute verification pipeline with authority weighting & conflicts | `draft_text`, `draft_file`, `source_files`, `source_authorities` (JSON) |
| `GET` | `/evaluation` | Benchmark evaluation metrics, F1 scores, and confusion matrix | `fresh` (bool) |
| `GET` | `/certificate/{id}` | Retrieve previously generated JSON certificate from disk | `certificate_id` |
| `GET` | `/certificate/{id}/report` | Retrieve human-readable Markdown verification report | `certificate_id` |
| `POST` | `/certificate/{id}/export` | Export downloadable JSON certificate | `certificate_id` |

---

## ⚖️ 11. Current Limitations

- **Scanned Images**: Pure image files (PNG/JPG/WEBP) currently output metadata with an explicit `"Image OCR support coming soon"` notification rather than synthesizing ungrounded text.
- **Encrypted Files**: Password-protected PDFs and Office documents require decryption before ingestion.
- **Unstructured Multi-Page Tables**: High-density nested tables in multi-column financial PDFs are parsed as sequential text lines; complex relational cross-cell joins may require tabular normalization.

---

## 🔮 12. Future Improvements

- **Native OCR Engine**: Integration with open-source OCR libraries (e.g., Tesseract or PaddleOCR) for scanned PDFs and embedded document images.
- **Citation Insertion**: Automatically injecting grounded inline footnote citations back into the original draft text.
- **Graph-Based Multi-Hop Reasoning**: Linking multi-document dependency chains where Claim A relies on Document X and Document Y simultaneously.
- **Batch Evaluation Pipeline**: Command-line interface for running bulk verification suites over enterprise document repositories.

---

## 🧪 13. Testing & Verification

Run the full automated test suite (36 tests across unit, integration, multi-format, authority weighting, and conflict detection):
```bash
pytest -v tests/
```

Run the benchmark evaluation runner (computes accuracy, precision, recall, F1, confusion matrix):
```bash
python evaluation/evaluate.py
```

Run the multi-format pipeline test (across all 9 supported formats):
```bash
python scripts/verify_multi_format_pipeline.py
```

Run the advanced verification demonstration (authority weighting + cross-source conflicts):
```bash
python scripts/demo_advanced_verification.py
```
