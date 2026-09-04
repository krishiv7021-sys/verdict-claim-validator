# ⚖️ VERDICT: Claim Validato

> **"A link is not proof."**
> A reproducible, evidence-grounded verification layer that ties every AI-generated claim to its exact source text.

**Team**: Cygnix  
**Track**: Generative AI & Trustworthy Systems  
**Project Type**: Hackathon MVP  

---

## 📌 Executive Summary

Large language models frequently generate convincing answers with subtle, dangerous factual hallucinations. Standard RAG applications or search interfaces only provide a list of search links or document references, leaving human reviewers to sift through 50-page PDFs to verify if a claim was actually substantiated.

**VERDICT: Claim Validato** turns verification into a **first-class, cryptographically anchored output**:
1. It ingests AI-generated drafts (TXT or pasted text) and ground-truth source documents across **9 common file formats**: `PDF`, `DOCX`, `PPTX`, `XLSX`, `CSV`, `TXT`, `MD`, `JSON`, `HTML`.
2. It hashes each source document using **SHA-256** to create an immutable audit trail.
3. It decomposes the draft into structured **atomic claims** ($C001, C002, \dots$).
4. It performs format-aware evidence retrieval preserving granular source locations (`Paragraph 32`, `Slide 7`, `Sheet!Row`, `Row 18`, `$.path`, `Page 4`).
5. It runs strict **entailment verification** against top candidate passages using high-precision LLMs (`groq`, `openai`, `gemini`) or a resilient local NLI engine.
6. It classifies each claim into one of three definitive verdicts:
   - **`SUPPORTED`** (Evidence directly confirms the claim)
   - **`REFUTED`** (Evidence directly contradicts numbers, dates, or meaning)
   - **`UNVERIFIED`** (Evidence is absent, insufficient, or ambiguous)
7. It issues a tamper-evident **Verification Certificate** (machine-readable JSON + human-readable audit report).

---

## 🏗️ Architecture & Verification Pipeline

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

## 🎯 The Three Verdicts

| Verdict | Definition | UI Treatment |
| :--- | :--- | :--- |
| **`SUPPORTED`** | The retrieved source evidence directly confirms the claim. | 🟢 Green badge, exact evidence snippet, page number, similarity score. |
| **`REFUTED`** | The retrieved source evidence directly contradicts the claim (e.g. numeric mismatch, opposite rule). | 🔴 Red badge, highlighted contradiction, requires reviewer sign-off. |
| **`UNVERIFIED`** | Source documents contain insufficient or ambiguous evidence. **Never fails upward into supported.** | 🟡 Yellow badge, explanation of missing coverage. |

### Conservative Document Verdict Logic
- If **any** claim is `REFUTED` $\rightarrow$ **`REVIEW_REQUIRED`**
- If **any** claim is `UNVERIFIED` $\rightarrow$ **`REVIEW_REQUIRED`**
- Only if **all** claims are `SUPPORTED` $\rightarrow$ **`VERIFIED`**

---

## 💻 Tech Stack

- **Frontend**: Streamlit with custom CSS, interactive claim selectors, and live workflow progress.
- **Backend API**: FastAPI, Pydantic v2, Uvicorn, Python Multipart.
- **Document Ingestion**: `pypdf`, UTF-8 / Latin-1 text chunking with page and character span tracking.
- **Embeddings**: `sentence-transformers` (`BAAI/bge-small-en-v1.5`) with resilient TF-IDF fallback.
- **Entailment**: Configurable multi-provider architecture (`groq`, `openai`, `gemini`) + rule-based heuristic NLI fallback.
- **Cryptographic Hashing**: Canonical `hashlib` SHA-256.
- **Testing & Benchmarking**: `pytest`, custom evaluation suite.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+ installed
- Virtual environment recommended

### 2. Installation
```bash
# Clone the repository
git clone <repo-url>
cd verdict-claim-validato

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Copy `.env.example` to `.env` (optional — works completely out of the box with built-in NLI engine):
```bash
cp .env.example .env
```

If you wish to use an LLM provider for entailment:
```env
# Example using Groq (free & ultra-fast)
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Or OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

---

## 🏃 Running the Application

### Option A: Run Backend & Frontend (Full Stack)
In Terminal 1 (Backend):
```bash
./venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

In Terminal 2 (Frontend):
```bash
./venv/bin/streamlit run frontend/app.py
```
Open your browser at `http://localhost:8501`.

### Option B: 1-Command Direct UI Launch
The Streamlit frontend automatically detects if the FastAPI server is running; if not, it seamlessly runs the verification pipeline in-process!
```bash
./venv/bin/streamlit run frontend/app.py
```

---

## 🧪 Testing & Evaluation

### Run Automated Tests
```bash
./venv/bin/pytest -v tests/
```
Tests cover:
- Cryptographic SHA-256 determinism and tamper detection
- PDF text extraction, page numbering, and paragraph chunking
- Compound sentence decomposition into atomic claims ($C001..$)
- End-to-end pipeline verification (Supported, Refuted, Unverified)
- Conservative verdict aggregation logic

### Run Benchmark Evaluation Suite
```bash
./venv/bin/python evaluation/evaluate.py
```
Outputs:
- Claim Verification Accuracy
- Top-K Evidence Retrieval Precision
- Per-class Precision, Recall, and F1-scores for `SUPPORTED`, `REFUTED`, and `UNVERIFIED`

---

## 📜 Machine-Readable Certificate Format

```json
{
  "certificate_version": "1.0",
  "certificate_id": "vc_e3a9c7b12d4f",
  "timestamp": "2026-09-04T02:00:00Z",
  "overall_verdict": "REVIEW_REQUIRED",
  "summary": {
    "total_claims": 4,
    "supported": 2,
    "refuted": 1,
    "unverified": 1
  },
  "configuration": {
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "entailment_provider": "heuristic_fallback",
    "top_k": 3,
    "similarity_threshold": 0.25
  },
  "sources": [
    {
      "source_id": "SRC_a982f1bc",
      "filename": "CRD_Compliance_Directive_2026.txt",
      "sha256": "4a72d3f9e8...",
      "file_type": "txt",
      "page_count": 1
    }
  ],
  "claims": [
    {
      "claim_id": "C001",
      "text": "Regulated entities are required to file their annual compliance disclosures within 30 days of the fiscal year close.",
      "verdict": "SUPPORTED",
      "confidence": 0.94,
      "reason": "Evidence directly confirms claim: '1.1 Mandatory Deadline: All regulated entities must complete and submit their annual compliance disclosures within 30 days...'",
      "retrieval_score": 0.88,
      "evidence": [
        {
          "source": "CRD_Compliance_Directive_2026.txt",
          "page": 1,
          "chunk_id": "SRC_a982f1bc-P1-C001",
          "text": "1.1 Mandatory Deadline: All regulated entities must complete and submit their annual compliance disclosures within 30 days following the conclusion of the fiscal year.",
          "start_char": 105,
          "end_char": 268,
          "similarity": 0.88
        }
      ]
    }
  ]
}
```

---

## ⚖️ Honest Limitations & Future Scope

### Stated Scope of this Hackathon MVP
- Verification is tailored to **text claims** grounded against **PDF and TXT documents**.
- Character offsets in PDFs depend on underlying text streams extracted via `pypdf`.

### Explicitly Out of Scope
- Multimodal / image / audio / video verification
- Formal mathematical theorem proving or code execution
- Automated rewriting of user drafts (VERDICT is an impartial verifier, not a generative writer)

### Future Roadmap
- Source freshness and domain authority weighting
- Table and structured tabular data extraction inside multi-column financial PDFs
- OCR integration for scanned raster PDFs
