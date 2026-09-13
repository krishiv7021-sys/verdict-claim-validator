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


def clean_single_claim(text: str) -> str:
    """Cleans a single claim string by removing leading labels, bullets, and enclosing quotes."""
    text = clean_text(text)
    # Strip leading label (e.g. "CLAIM 1:", "Claim 2 -", "1.", "-", "*", "•")
    text = re.sub(
        r'^(?:(?:CLAIM|FACT|ASSERTION|STATEMENT|POINT|CASE|ITEM)\s*\d+[:\-\.]?\s*|[0-9]+[\.\)]\s*|[\-\*•]\s*)',
        '',
        text,
        flags=re.IGNORECASE
    ).strip()
    # Strip enclosing quotes
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")) or \
       (text.startswith('“') and text.endswith('”')):
        text = text[1:-1].strip()
    # Ensure standard trailing punctuation
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    return text


def decompose_claims_list(claims_list: List[str]) -> List[AtomicClaim]:
    """
    Directly converts a list of user-provided claim strings into structured AtomicClaim objects.
    Assigns sequential Claim IDs (C001, C002, ...) and ensures thorough cleaning.
    """
    claims: List[AtomicClaim] = []
    claim_counter = 1

    for c in claims_list:
        c_str = str(c).strip()
        if not c_str:
            continue
        cleaned = clean_single_claim(c_str)
        if len(cleaned.strip('.').strip()) <= 3:
            continue
        claim_id = f"C{claim_counter:03d}"
        claims.append(
            AtomicClaim(
                claim_id=claim_id,
                text=cleaned,
                original_sentence=c_str
            )
        )
        claim_counter += 1

    return claims


def split_draft_into_sentences(draft_text: str) -> List[str]:
    """
    Splits draft text into grammatically valid sentences without breaking on known abbreviations.
    Handles bullet points, numbering, claim headers, JSON arrays, quoted lists, and paragraph structure cleanly.
    """
    text_strip = draft_text.strip()
    if not text_strip:
        return []

    # 1. JSON / Python list representation (e.g. ["claim1", "claim2"] or claims = [...])
    list_match = re.match(r'^(?:claims\s*=\s*)?(\[.*\])$', text_strip, re.DOTALL)
    if list_match:
        try:
            import json
            parsed = json.loads(list_match.group(1))
            if isinstance(parsed, list):
                extracted = [str(x).strip() for x in parsed if str(x).strip()]
                if extracted:
                    return extracted
        except Exception:
            try:
                import ast
                parsed = ast.literal_eval(list_match.group(1))
                if isinstance(parsed, list):
                    extracted = [str(x).strip() for x in parsed if str(x).strip()]
                    if extracted:
                        return extracted
            except Exception:
                pass

    # 2. Explicit Claim / Fact labels (e.g. Claim 1: "...", Claim 2: "...")
    claim_pat = re.compile(
        r'(?:CLAIM|FACT|ASSERTION)\s*\d+[:\-\.]?\s*[\n\r]*\s*[\"“\']?([^\"”\'\n\r]+)[\"“\']?',
        re.IGNORECASE
    )
    labeled = claim_pat.findall(text_strip)
    if len(labeled) >= 2:
        extracted = [c.strip() for c in labeled if len(c.strip()) > 5]
        if extracted:
            return extracted

    # 3. Multiple quoted spans on the same line or block (e.g. "claim 1" "claim 2" or "claim 1", "claim 2")
    quotes = re.findall(r'[\"“\']([^\"”\'\n\r]{8,})[\"“\']', text_strip)
    if len(quotes) >= 2:
        quoted_len = sum(len(q) for q in quotes)
        if quoted_len / len(text_strip) >= 0.50:
            return [q.strip() for q in quotes if q.strip()]

    # 4. Standard line & sentence splitting
    lines = [line.strip() for line in draft_text.splitlines() if line.strip()]
    merged_lines: List[str] = []
    pending_header = ""

    for line in lines:
        # Check if line is a standalone label/header (e.g., "CLAIM 1:", "Claim 2", "Fact 1:")
        if re.match(r'^(?:CLAIM|FACT|ASSERTION|STATEMENT|POINT|CASE|ITEM)\s*\d+[:\-\.]?$', line, re.IGNORECASE):
            pending_header = line
            continue
        # Skip standalone document headers (e.g. "DOCUMENT 1:", "SOURCE A:")
        if re.match(r'^(?:DOCUMENT|DOC|SOURCE|FILE|EVIDENCE)\s*\d+[:\-\.]?$', line, re.IGNORECASE):
            continue
        if pending_header:
            line = f"{pending_header} {line}"
            pending_header = ""
        merged_lines.append(line)

    raw_sentences = []

    for line in merged_lines:
        # Strip leading claim labels, bullets, or numbering
        line_clean = re.sub(
            r'^(?:(?:CLAIM|FACT|ASSERTION|STATEMENT|POINT|CASE|ITEM)\s*\d+[:\-\.]?\s*|[0-9]+[\.\)]\s*|[\-\*•]\s*)',
            '',
            line,
            flags=re.IGNORECASE
        ).strip()

        # Strip enclosing quotation marks
        if (line_clean.startswith('"') and line_clean.endswith('"')) or \
           (line_clean.startswith("'") and line_clean.endswith("'")) or \
           (line_clean.startswith('“') and line_clean.endswith('”')):
            line_clean = line_clean[1:-1].strip()

        if not line_clean:
            continue

        # Split on sentence ending punctuation followed by space (or quote + space)
        potential_splits = re.split(
            r'(?:(?<=[\.\?!])|(?<=[\.\?!][\"“\']))\s*(?:,\s*)?(?=[A-Z0-9\$\"“\'\(])',
            line_clean
        )
        
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

    # Pattern: Semicolon separated independent assertions
    if ';' in sentence:
        semi_parts = [p.strip() for p in sentence.split(';') if p.strip()]
        if len(semi_parts) >= 2 and all(len(p.split()) >= 4 for p in semi_parts):
            return semi_parts
    
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
    1. Splits text into sentences or extracts explicit claim elements.
    2. Breaks compound sentences into atomic claims.
    3. Assigns unique Claim IDs (C001, C002, ...).
    4. Preserves association with the original draft sentence.
    """
    text_clean = (draft_text or "").strip()
    if not text_clean:
        return []

    # Check for direct JSON or Python list input
    list_match = re.match(r'^(?:claims\s*=\s*)?(\[.*\])$', text_clean, re.DOTALL)
    if list_match:
        try:
            import json
            parsed = json.loads(list_match.group(1))
            if isinstance(parsed, list):
                extracted = [str(x).strip() for x in parsed if str(x).strip()]
                if extracted:
                    return decompose_claims_list(extracted)
        except Exception:
            try:
                import ast
                parsed = ast.literal_eval(list_match.group(1))
                if isinstance(parsed, list):
                    extracted = [str(x).strip() for x in parsed if str(x).strip()]
                    if extracted:
                        return decompose_claims_list(extracted)
            except Exception:
                pass

    sentences = split_draft_into_sentences(draft_text)
    claims: List[AtomicClaim] = []
    claim_counter = 1

    for sentence in sentences:
        atomic_texts = decompose_compound_sentence(sentence)
        for a_text in atomic_texts:
            cleaned = clean_single_claim(a_text)
            if len(cleaned.strip('.').strip()) <= 3:
                continue

            claim_id = f"C{claim_counter:03d}"
            claims.append(
                AtomicClaim(
                    claim_id=claim_id,
                    text=cleaned,
                    original_sentence=sentence
                )
            )
            claim_counter += 1

    return claims
