import re
from typing import Dict, Any

SAMPLE_DEMO_SOURCE_TEXT = """CORPORATE REGULATORY COMPLIANCE DIRECTIVE (CRD)
Document Reference: CRD-SEC-V2
Effective Date: January 1

SECTION 1: FILING REQUIREMENTS & TIMELINES
1.1 Mandatory Deadline: All regulated entities must complete and submit their annual compliance disclosures within 30 days following the conclusion of the fiscal year.
1.2 Electronic Submission: All annual filings must be transmitted exclusively through the secure electronic regulatory gateway (EDGAR-Plus). Physical paper submissions are prohibited.
1.3 Executive Attestation: The Chief Executive Officer and Chief Compliance Officer must jointly sign the digital attestation certificate.

SECTION 2: PENALTIES & ENFORCEMENT
2.1 Late Filing Penalties: Any entity failing to meet the 30-day submission deadline is subject to a statutory base civil penalty of $15,000 per violation, plus $500 for each subsequent day of delay.
2.2 Willful Misrepresentation: Knowingly submitting falsified compliance records constitutes a Class B regulatory felony subject to license suspension.

SECTION 3: EXEMPTIONS & SPECIAL PROVISIONS
3.1 Small Enterprise Exemption: Commercial enterprises with an aggregate gross annual revenue of less than $2.5 million are explicitly exempt from mandatory quarterly third-party audit verifications.
3.2 Extension Requests: A single 14-day hardship extension may be granted upon formal written notice submitted at least 5 business days prior to the statutory deadline.
"""

SAMPLE_DEMO_AI_DRAFT = """Here is the executive briefing based on the compliance directive:

1. Regulated entities are required to file their annual compliance disclosures within 30 days of the fiscal year close.
2. The policy mandates that all annual filings must be submitted electronically.
3. The statutory base penalty for late filing violations is $50,000 per violation.
4. All regulated organizations are strictly required to conduct bi-monthly climate risk disclosure audits."""


def get_demo_package() -> Dict[str, Any]:
    """Returns bundled demo draft and source document for 1-click system demonstrations."""
    return {
        "draft_text": SAMPLE_DEMO_AI_DRAFT,
        "source_filename": "CRD_Compliance_Directive.txt",
        "source_content": SAMPLE_DEMO_SOURCE_TEXT
    }


def truncate_text(text: str, max_chars: int = 150) -> str:
    """Safely truncates string with ellipsis."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."
