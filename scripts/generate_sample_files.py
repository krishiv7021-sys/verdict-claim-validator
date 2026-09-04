import os
from pathlib import Path

# Create data directory if not exists
data_dir = Path(__file__).parent.parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)

# 1. Sample TXT file
txt_path = data_dir / "sample_policy.txt"
txt_content = """CORPORATE COMPLIANCE AND REGULATORY STANDARD (CCRS-2026)
Effective Date: January 1, 2026

1. FILING OBLIGATIONS:
All commercial entities must complete their corporate filings within 30 days of the fiscal year close.
Submissions must be delivered exclusively via the digital reporting system.

2. PENALTIES:
The statutory civil penalty for late filings is $15,000 per violation.
Willful misrepresentation is subject to immediate license suspension.

3. AUDIT PROVISIONS:
Enterprises generating under $2.5 million in revenue are exempt from mandatory quarterly audits.
"""

with open(txt_path, "w", encoding="utf-8") as f:
    f.write(txt_content)

print(f"Created sample TXT: {txt_path}")

# 2. Sample PDF file
pdf_path = data_dir / "sample_policy.pdf"

# Raw PDF 1.4 specification stream
pdf_stream_text = (
    "BT\n"
    "/F1 14 Tf\n"
    "50 720 Td\n"
    "(CORPORATE REGULATORY COMPLIANCE DIRECTIVE 2026) Tj\n"
    "0 -30 Td\n"
    "/F1 11 Tf\n"
    "(Section 1: All corporate annual filings must be completed within 30 days.) Tj\n"
    "0 -20 Td\n"
    "(Section 2: Filings must be transmitted through the electronic portal.) Tj\n"
    "0 -20 Td\n"
    "(Section 3: The statutory base penalty for late filing is \\$15,000 per violation.) Tj\n"
    "0 -20 Td\n"
    "(Section 4: Entities with revenue under \\$2.5 million are exempt from quarterly audits.) Tj\n"
    "ET\n"
)

stream_len = len(pdf_stream_text.encode("latin-1"))

pdf_body = f"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length {stream_len} >>
stream
{pdf_stream_text}endstream
endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
0000000000 65535 f 
0000000010 00000 n 
0000000060 00000 n 
0000000117 00000 n 
0000000244 00000 n 
0000000600 00000 n 
trailer << /Size 6 /Root 1 0 R >>
startxref
680
%%EOF
"""

with open(pdf_path, "wb") as f:
    f.write(pdf_body.encode("latin-1"))

print(f"Created sample PDF: {pdf_path}")

# Verify extraction via pypdf
import pypdf
reader = pypdf.PdfReader(str(pdf_path))
extracted = reader.pages[0].extract_text()
print("PDF Extraction Verified:")
print(extracted.strip())
