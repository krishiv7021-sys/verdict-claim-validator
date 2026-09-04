from backend.services.claim_decomposer import (
    split_draft_into_sentences,
    decompose_compound_sentence,
    decompose_draft
)


def test_split_sentences_with_abbreviations():
    text = "Dr. Smith filed the document on Jan. 15. The company was Inc. registered in Delaware."
    sentences = split_draft_into_sentences(text)
    assert len(sentences) == 2
    assert "Dr. Smith" in sentences[0]


def test_decompose_compound_sentence():
    compound = "The policy requires companies to file within 30 days and submit the report electronically."
    atomic = decompose_compound_sentence(compound)
    assert len(atomic) == 2
    assert "30 days" in atomic[0]
    assert "electronically" in atomic[1]


def test_decompose_draft_pipeline():
    draft = """1. Filings are due within 30 days.
2. The company must submit electronically.
3. Penalty is $10,000."""
    claims = decompose_draft(draft)
    assert len(claims) >= 3
    assert claims[0].claim_id == "C001"
    assert claims[1].claim_id == "C002"
    assert claims[2].claim_id == "C003"
    assert claims[0].original_sentence is not None
