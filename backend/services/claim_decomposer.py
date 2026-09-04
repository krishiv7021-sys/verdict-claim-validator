import re
from typing import List
from backend.schemas import AtomicClaim


ABBREVIATIONS = {
    "dr.", "mr.", "mrs.", "ms.", "inc.", "corp.", "ltd.", "co.", "e.g.", "i.e.",
    "etc.", "vs.", "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.",
    "sep.", "sept.", "oct.", "nov.", "dec.", "sec.", "art.", "no.", "fig."
}


def clean_text(text: str) -> str:
    """Removes excessive whitespace and standardizes quotes/dashes."""
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('“', '"').replace('”', '"').replace('’', "'")
    return text.strip()


def split_draft_into_sentences(draft_text: str) -> List[str]:
    """
    Splits draft text into grammatically valid sentences without breaking on known abbreviations.
    Handles bullet points, numbering, and paragraph structure cleanly.
    """
    # Normalize newlines
    lines = [line.strip() for line in draft_text.splitlines() if line.strip()]
    raw_sentences = []

    for line in lines:
        # Strip leading bullet indicators (e.g. "1.", "-", "*", "•")
        line_clean = re.sub(r'^(?:[0-9]+[\.\)]|[\-\*•])\s*', '', line).strip()
        if not line_clean:
            continue

        # Split on sentence ending punctuation followed by space
        potential_splits = re.split(r'(?<=[\.\?!])\s+(?=[A-Z0-9\$\"\'\(])', line_clean)
        
        merged: List[str] = []
        for part in potential_splits:
            part_clean = part.strip()
            if not part_clean:
                continue
            
            # Check if previous part ended with an abbreviation
            if merged:
                last_word = merged[-1].split()[-1].lower() if merged[-1].split() else ""
                if last_word in ABBREVIATIONS or len(last_word) <= 2:
                    merged[-1] = merged[-1] + " " + part_clean
                    continue
            
            merged.append(part_clean)
        
        raw_sentences.extend(merged)

    return [s for s in raw_sentences if len(s.strip()) > 5]


def decompose_compound_sentence(sentence: str) -> List[str]:
    """
    Decomposes compound sentences into atomic, testable claims when safe to do so.
    Preserves subject and context. Does not split if split loses semantic completeness.
    """
    sentence = clean_text(sentence)
    
    # Pattern: "Subject Verb ... and [also|additionally] Verb ..."
    # E.g. "The policy requires companies to file within 30 days and submit the report electronically."
    match_and = re.search(
        r'^(?P<subject>.+?\b(?:requires|mandates|obligates|states that|specifies that|must|should|shall))\s+(?P<clause1>.+?)\s+and\s+(?:also\s+)?(?P<clause2>(?:submit|file|pay|provide|include|report|disclose|adhere|notify).+)$',
        sentence,
        re.IGNORECASE
    )
    if match_and:
        subj = match_and.group("subject").strip()
        c1 = match_and.group("clause1").strip()
        c2 = match_and.group("clause2").strip()
        claim1 = f"{subj} {c1}."
        claim2 = f"{subj} {c2}."
        return [claim1, claim2]

    # Pattern: "..., but ..." or "..., while ..."
    contrast_match = re.search(r'^(.+?),\s*(?:while|whereas|however)\s*(.+)$', sentence, re.IGNORECASE)
    if contrast_match:
        c1 = contrast_match.group(1).strip()
        c2 = contrast_match.group(2).strip()
        if len(c1.split()) >= 4 and len(c2.split()) >= 4:
            if not c1.endswith('.'):
                c1 += '.'
            if not c2.endswith('.'):
                c2 += '.'
            # Capitalize first letter of c2
            c2 = c2[0].upper() + c2[1:]
            return [c1, c2]

    return [sentence]


def decompose_draft(draft_text: str) -> List[AtomicClaim]:
    """
    Main claim decomposition pipeline:
    1. Splits text into sentences.
    2. Breaks compound sentences into atomic claims.
    3. Assigns unique Claim IDs (C001, C002, ...).
    4. Preserves association with the original draft sentence.
    """
    sentences = split_draft_into_sentences(draft_text)
    claims: List[AtomicClaim] = []
    claim_counter = 1

    for sentence in sentences:
        atomic_texts = decompose_compound_sentence(sentence)
        for a_text in atomic_texts:
            a_text = clean_text(a_text)
            if not a_text.endswith(('.', '!', '?')):
                a_text += '.'

            claim_id = f"C{claim_counter:03d}"
            claims.append(
                AtomicClaim(
                    claim_id=claim_id,
                    text=a_text,
                    original_sentence=sentence
                )
            )
            claim_counter += 1

    return claims
