import os
import json
import logging
import re
from typing import List, Tuple, Dict, Any
import requests

from backend.schemas import VerdictType, EvidenceSpan

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert impartial evidence-grounded verification engine.
Your task is to verify whether an AI-generated atomic claim is supported, refuted, or unverified by the provided source evidence.

STRICT VERDICT RULES:
1. "SUPPORTED": The provided evidence directly and explicitly confirms the truth of the claim.
2. "REFUTED": The provided evidence directly contradicts or negates the claim (e.g. different dates, numbers, requirements, or opposite meaning).
3. "UNVERIFIED": The evidence does not contain sufficient information to prove or disprove the claim, or the evidence is absent/ambiguous.

CRITICAL PRINCIPLE:
- Never turn uncertainty into SUPPORTED.
- If evidence is weak, partial, or ambiguous, return "UNVERIFIED".
- You must output ONLY a valid JSON object with keys: "verdict", "confidence", and "reason".
"""

def extract_json_from_response(text: str) -> Dict[str, Any]:
    """Safely extracts JSON from an LLM response."""
    text = text.strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {"verdict": "UNVERIFIED", "confidence": 0.0, "reason": "Failed to parse model output."}


class HeuristicNLIEntailmentEngine:
    """
    High-precision, deterministic fallback entailment engine.
    Analyzes numeric discrepancies, modal verbs, negation polarity, and semantic overlap.
    Guarantees zero crashes and conservative UNVERIFIED outputs when uncertain.
    """
    def verify_claim(self, claim_text: str, evidence_spans: List[EvidenceSpan]) -> Tuple[VerdictType, float, str]:
        if not evidence_spans:
            return (
                VerdictType.UNVERIFIED,
                0.0,
                "No relevant evidence was found in the uploaded source documents."
            )

        # Use top candidate evidence
        best_evidence = evidence_spans[0]
        ev_text = best_evidence.text.strip()
        c_text = claim_text.strip()

        # If similarity is very low, do not attempt to verify
        if best_evidence.similarity < 0.35:
            return (
                VerdictType.UNVERIFIED,
                round(best_evidence.similarity, 2),
                f"Closest retrieved passage in {best_evidence.source} has insufficient semantic relevance (similarity {best_evidence.similarity:.2f})."
            )

        # 1. Number / Metric Contradiction Detection
        def _extract_metrics(text: str):
            t = text.lower()
            # Remove comma separators in numbers (e.g. 90,000 -> 90000)
            t = re.sub(r'(\d),(\d)', r'\1\2', t)
            matches = list(re.finditer(r'(\$)?\s*(\d+(?:\.\d+)?)\s*(million|billion|days?|months?|years?|dollars?|%)?', t))
            metrics = []
            for m in matches:
                is_cur, num_str, unit = m.groups()
                if not num_str:
                    continue
                val = float(num_str)
                unit_norm = ""
                if is_cur or unit == "dollars":
                    unit_norm = "currency"
                elif unit in ("million", "billion"):
                    val *= 1_000_000 if unit == "million" else 1_000_000_000
                    unit_norm = "scaled"
                elif unit:
                    unit_norm = unit.rstrip('s')
                metrics.append((val, unit_norm, m.group(0).strip()))
            return metrics

        c_metrics = _extract_metrics(c_text)
        ev_metrics = _extract_metrics(ev_text)

        loc_str = best_evidence.location or (f"Page {best_evidence.page}" if best_evidence.page else "Source Document")

        if c_metrics and ev_metrics:
            for c_val, c_unit, c_raw in c_metrics:
                for ev_val, ev_unit, ev_raw in ev_metrics:
                    # Same unit category (or both dimensionless numbers) but different values
                    if (c_unit == ev_unit or (not c_unit and not ev_unit)) and abs(c_val - ev_val) > 0.01:
                        return (
                            VerdictType.REFUTED,
                            0.95,
                            f"Direct numeric contradiction: Claim specifies '{c_raw}' but source document specifies '{ev_raw}' in {best_evidence.source} ({loc_str})."
                        )

        # 2. Negation & Permission Contradiction Detection
        c_tokens = set(re.findall(r'\b[a-z]+\b', c_text.lower()))
        ev_tokens = set(re.findall(r'\b[a-z]+\b', ev_text.lower()))

        negations = {"not", "never", "cannot", "neither", "nor"}
        c_has_neg = bool(c_tokens.intersection(negations))
        ev_has_neg = bool(ev_tokens.intersection(negations))

        allow_words = {"accepted", "allowed", "permitted", "permissible", "voluntary", "optional"}
        prohibit_words = {"prohibited", "banned", "forbidden", "disallowed"}

        c_allows = bool(c_tokens.intersection(allow_words))
        ev_prohibits = bool(ev_tokens.intersection(prohibit_words))
        c_prohibits = bool(c_tokens.intersection(prohibit_words))
        ev_allows = bool(ev_tokens.intersection(allow_words))

        # 3. Content overlap with lightweight suffix normalization
        def _stem(w: str) -> str:
            for sfx in ('ing', 'ed', 'es', 's', 'tion', 'ment', 'ies'):
                if w.endswith(sfx) and len(w) > len(sfx) + 2:
                    if sfx == 'ies':
                        return w[:-3] + 'y'
                    return w[:-len(sfx)]
            return w

        stopwords = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "in", "that", "it", "for", "on", "with", "all", "must", "be", "their", "of"}
        c_keywords = {_stem(t) for t in (c_tokens - stopwords - negations - allow_words - prohibit_words)}
        ev_keywords = {_stem(t) for t in (ev_tokens - stopwords - negations - allow_words - prohibit_words)}

        if not c_keywords:
            return (VerdictType.UNVERIFIED, 0.0, "Claim lacks verifiable factual keywords.")

        overlap = c_keywords.intersection(ev_keywords)
        overlap_ratio = len(overlap) / len(c_keywords)

        # Contradiction Case A: Claim says allowed/accepted, but source says prohibited (or vice versa)
        if (c_allows and ev_prohibits and overlap_ratio >= 0.35) or (c_prohibits and ev_allows and overlap_ratio >= 0.35):
            return (
                VerdictType.REFUTED,
                0.92,
                f"Direct regulatory contradiction: Claim asserts permission/prohibition contradicted by source in {best_evidence.source} ({loc_str})."
            )

        # Contradiction Case B: Claim explicitly negates what source affirms
        if c_has_neg and not ev_has_neg and overlap_ratio >= 0.60:
            return (
                VerdictType.REFUTED,
                0.90,
                f"Contradictory polarity: Claim negates source requirement in {best_evidence.source} ({loc_str})."
            )

        # Support criteria:
        # High keyword overlap and strong retrieval similarity
        if overlap_ratio >= 0.45 and best_evidence.similarity >= 0.50:
            return (
                VerdictType.SUPPORTED,
                round(min(0.96, best_evidence.similarity + 0.1), 2),
                f"Evidence directly confirms claim: '{ev_text[:120]}...' in {best_evidence.source} ({loc_str})."
            )
        elif overlap_ratio >= 0.35 and best_evidence.similarity >= 0.70:
            return (
                VerdictType.SUPPORTED,
                round(min(0.92, best_evidence.similarity), 2),
                f"Evidence confirms claim meaning in {best_evidence.source} ({loc_str})."
            )

        # Conservative fallback
        return (
            VerdictType.UNVERIFIED,
            round(best_evidence.similarity, 2),
            f"Evidence in {best_evidence.source} ({loc_str}) is related but does not fully establish or contradict the claim."
        )


class LLMEntailmentEngine:
    """
    LLM-backed entailment engine using Groq, OpenAI, or Gemini via unified HTTP interface.
    Falls back to HeuristicNLIEntailmentEngine on network timeout or API error.
    """
    def __init__(self, provider: str, api_key: str, model_name: str):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model_name = model_name
        self.fallback = HeuristicNLIEntailmentEngine()

    def verify_claim(self, claim_text: str, evidence_spans: List[EvidenceSpan]) -> Tuple[VerdictType, float, str]:
        if not evidence_spans:
            return (VerdictType.UNVERIFIED, 0.0, "No evidence passages retrieved.")

        # Prepare evidence text summary
        evidence_context = "\n---\n".join([
            f"[Source: {e.source}, Page {e.page} (Similarity: {e.similarity:.2f})]:\n{e.text}"
            for e in evidence_spans[:3]
        ])

        user_prompt = f"""EVIDENCE:
{evidence_context}

CLAIM TO VERIFY:
"{claim_text}"

Respond with a JSON object:
{{
    "verdict": "SUPPORTED" | "REFUTED" | "UNVERIFIED",
    "confidence": <float between 0.0 and 1.0>,
    "reason": "<one concise sentence explaining verdict>"
}}"""

        try:
            if self.provider == "groq":
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model_name or "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"}
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=12)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    parsed = extract_json_from_response(content)
                    return self._process_verdict(parsed)

            elif self.provider == "openai":
                url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model_name or "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"}
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=12)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    parsed = extract_json_from_response(content)
                    return self._process_verdict(parsed)

            elif self.provider == "gemini":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name or 'gemini-1.5-flash'}:generateContent?key={self.api_key}"
                headers = {"Content-Type": "application/json"}
                payload = {
                    "contents": [{
                        "parts": [{"text": f"{SYSTEM_PROMPT}\n\n{user_prompt}"}]
                    }],
                    "generationConfig": {
                        "response_mime_type": "application/json",
                        "temperature": 0.0
                    }
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=12)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["candidates"][0]["content"]["parts"][0]["text"]
                    parsed = extract_json_from_response(content)
                    return self._process_verdict(parsed)

        except Exception as e:
            logger.warning(f"LLM entailment request failed ({e}). Falling back to internal engine.")

        # Graceful fallback to heuristic NLI engine
        return self.fallback.verify_claim(claim_text, evidence_spans)

    def _process_verdict(self, parsed: Dict[str, Any]) -> Tuple[VerdictType, float, str]:
        raw_v = str(parsed.get("verdict", "")).strip().upper()
        conf = float(parsed.get("confidence", 0.8))
        reason = str(parsed.get("reason", "")).strip()

        if raw_v == "SUPPORTED":
            verdict = VerdictType.SUPPORTED
        elif raw_v == "REFUTED":
            verdict = VerdictType.REFUTED
        else:
            verdict = VerdictType.UNVERIFIED

        return (verdict, round(conf, 2), reason)


def get_entailment_engine():
    """
    Factory creating configured entailment engine based on environment variables.
    Defaults to HeuristicNLIEntailmentEngine if no API key is set.
    """
    provider = os.getenv("LLM_PROVIDER", "").lower()
    groq_key = os.getenv("GROQ_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")

    if provider == "groq" and groq_key:
        return LLMEntailmentEngine("groq", groq_key, os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    elif provider == "openai" and openai_key:
        return LLMEntailmentEngine("openai", openai_key, os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    elif provider == "gemini" and gemini_key:
        return LLMEntailmentEngine("gemini", gemini_key, os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
    elif groq_key:
        return LLMEntailmentEngine("groq", groq_key, "llama-3.3-70b-versatile")
    elif openai_key:
        return LLMEntailmentEngine("openai", openai_key, "gpt-4o-mini")
    elif gemini_key:
        return LLMEntailmentEngine("gemini", gemini_key, "gemini-1.5-flash")

    return HeuristicNLIEntailmentEngine()
