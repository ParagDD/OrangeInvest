"""Untrusted-text handling and trace redaction. Free text from tools is DATA, never instructions."""
import re
from dataclasses import dataclass

_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
INJECTION_PATTERNS = [re.compile(p, re.I) for p in [
    r"ignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+(instructions|rules|messages)",
    r"disregard\s+(the\s+)?(previous|above|system)",
    r"\byou\s+are\s+now\b",
    r"\badmin(istrator)?\s+mode\b",
    r"\bsystem\s*(prompt|message|:)",
    r"\bwithout\s+(asking|confirmation|approval)\b",
    r"\bauto[-\s]?(confirm|approve|book)\b",
    r"\b(cancel|delete)\s+(all|every)\b",
    r"\bset\s+the\s+price\b",
    r"\bnew\s+instructions?\b",
    r"\bassistant\s*:",
]]
_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_CLAIM = re.compile(r"\b(booked|confirmed|cancell?ed|approved|rescheduled)\b", re.I)
_SECRET_KEY = re.compile(r"(api[_-]?key|secret|token|password|authorization|credential)", re.I)
_SECRET_VAL = re.compile(r"(sk-[A-Za-z0-9_\-]{16,}|Bearer\s+[A-Za-z0-9._\-]{16,})")


@dataclass(frozen=True)
class Untrusted:
    text: str
    flagged: bool
    source: str = "review"


def looks_like_injection(s: str) -> bool:
    return any(p.search(s) for p in INJECTION_PATTERNS)


def sanitize_text(s, max_len=300, source="review") -> Untrusted:
    s = _CTRL.sub(" ", str(s))
    s = re.sub(r"\s+", " ", s).strip()
    flagged = looks_like_injection(s)
    return Untrusted(s[:max_len], flagged, source)


def neutralize_note(s: str, max_len=240) -> str:
    """Model-written text derived from untrusted input is shown only as a labelled, unverified note.
    It may not carry links or claims of booking/confirmation: those come from code only."""
    s = _URL.sub("[link removed]", str(s))
    s = _CLAIM.sub("[…]", s)
    return re.sub(r"\s+", " ", s).strip()[:max_len]


def redact(obj, max_str=200):
    if isinstance(obj, dict):
        return {k: ("[REDACTED]" if _SECRET_KEY.search(str(k)) else redact(v, max_str)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [redact(v, max_str) for v in obj]
    if isinstance(obj, str):
        s = _SECRET_VAL.sub("[REDACTED]", obj)
        return s if len(s) <= max_str else s[:max_str] + "…"
    return obj
