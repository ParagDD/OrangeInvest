"""Model interface. The model only turns language into a *validated, closed-schema* Intent and writes
quarantined notes. It cannot approve anything: there is no 'confirm' intent kind, approvals are created by
the chat layer from the user's raw message. It never sees tool free-text in interpret()."""
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

SERVICES = {"plumber", "electrician", "ac_service"}
KNOWN_AREAS = ["Baner", "Koregaon Park", "Kothrud", "Viman Nagar"]
SLOT_KEYS = {"service", "area", "date_from", "date_to", "earliest_time", "latest_time", "budget",
             "provider_pref", "technician_pref", "target_booking"}
MEMORY_KEYS = {"area", "earliest_time", "preferred_technician"}
KINDS = {"task", "slots", "memory", "decline", "smalltalk", "unknown"}
ACTIONS = {"book", "cancel", "reschedule"}
MEMORY_OPS = {"view", "set", "forget", "confirm", "reject"}

# High-impact intents must be grounded in the user's own words (defence in depth against a hijacked model).
VERBS = {
    "cancel": r"\b(cancel|drop|remove|delete|call off)\b",
    "reschedule": r"\b(move|reschedule|shift|postpone|change|push|rebook)\b",
    "book": r"\b(book|find|need|get|want|schedule|hire|arrange|looking|fix|repair|service|same|another)\b",
}


@dataclass
class Intent:
    kind: str = "unknown"
    action: Optional[str] = None
    slots: dict = field(default_factory=dict)
    clear: list = field(default_factory=list)
    memory: Optional[dict] = None


def _hhmm(v):
    return isinstance(v, str) and re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", v) is not None


def _iso(v):
    try:
        date.fromisoformat(v)
        return True
    except Exception:
        return False


_SLOT_VALIDATORS = {
    "service": lambda v: v in SERVICES,
    "area": lambda v: isinstance(v, str) and re.fullmatch(r"[A-Za-z .'\-]{1,60}", v) is not None,
    "date_from": lambda v: isinstance(v, str) and _iso(v),
    "date_to": lambda v: isinstance(v, str) and _iso(v),
    "earliest_time": _hhmm,
    "latest_time": _hhmm,
    "budget": lambda v: isinstance(v, int) and not isinstance(v, bool) and 100 <= v <= 100000,
    "provider_pref": lambda v: isinstance(v, str) and (v == "last" or re.fullmatch(r"p_[a-z0-9_]+", v) is not None),
    "technician_pref": lambda v: isinstance(v, str) and re.fullmatch(r"[A-Za-z .'\-]{1,40}", v) is not None,
    "target_booking": lambda v: isinstance(v, str) and re.fullmatch(r"B-\d+", v) is not None,
}


def validate_intent(raw):
    """Returns (Intent, None) or (None, reason). Anything outside the closed schema is rejected."""
    if not isinstance(raw, Intent):
        return None, "not an Intent"
    if raw.kind not in KINDS:
        return None, f"unknown kind {raw.kind!r}"
    if raw.action is not None and raw.action not in ACTIONS:
        return None, f"unknown action {raw.action!r}"
    if not isinstance(raw.slots, dict):
        return None, "slots must be a dict"
    for k, v in raw.slots.items():
        if k not in SLOT_KEYS:
            return None, f"slot {k!r} not allowed"
        if not _SLOT_VALIDATORS[k](v):
            return None, f"invalid value for {k}"
    if any(k not in SLOT_KEYS for k in raw.clear):
        return None, "clear contains unknown key"
    if raw.kind == "task" and raw.action is None:
        return None, "task without action"
    if raw.memory is not None:
        m = raw.memory
        if m.get("op") not in MEMORY_OPS or (m.get("key") is not None and m["key"] not in MEMORY_KEYS):
            return None, "bad memory op"
        if m.get("op") == "set":
            v = m.get("value")
            if not isinstance(v, str) or not v or (m["key"] == "earliest_time" and not _hhmm(v)) or len(v) > 60:
                return None, "bad memory value"
    return raw, None


def grounded(intent: Intent, text: str) -> bool:
    if intent.kind == "task" and intent.action in VERBS:
        return re.search(VERBS[intent.action], text, re.I) is not None
    return True


# -- scripted model (tests) --------------------------------------------------------------------------
class ScriptExhausted(AssertionError):
    pass


class ScriptedModel:
    """Replays a recorded sequence of Intents. Deterministic, offline, free."""
    last_usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

    def __init__(self):
        self.queue, self.calls = [], []

    def push(self, intent):
        self.queue.append(intent)

    def interpret(self, text, ctx):
        self.calls.append({"fn": "interpret", "text": text, "ctx": ctx})
        if not self.queue:
            raise ScriptExhausted(f"model called with no scripted intent for: {text!r}")
        return self.queue.pop(0)

    def summarize_reviews(self, reviews):
        self.calls.append({"fn": "summarize_reviews", "reviews": reviews})
        return {pid: "Customers describe reliable, on-time work." for pid in reviews}


# -- rule-based model (CLI demo without an LLM) --------------------------------------------------------
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


class RuleModel:
    last_usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

    def interpret(self, text, ctx):
        t = text.lower()
        today = date.fromisoformat(ctx["today"])
        # memory
        if re.search(r"what do you (remember|know)|my (saved )?preferences|what.*saved", t):
            return Intent("memory", memory={"op": "view"})
        m = re.search(r"forget (?:my |the )?(area|earliest|technician|time)", t)
        if m:
            key = {"area": "area", "earliest": "earliest_time", "time": "earliest_time", "technician": "preferred_technician"}[m.group(1)]
            return Intent("memory", memory={"op": "forget", "key": key})
        m = re.search(r"remember ([A-Za-z]+)", t)
        if m and not re.search(r"remember (that|my)", t):
            return Intent("memory", memory={"op": "confirm", "key": "preferred_technician"})
        m = re.search(r"(?:my area is|i live in|change my area to)\s+([A-Za-z ]+)", text, re.I)
        if m:
            return Intent("memory", memory={"op": "set", "key": "area", "value": m.group(1).strip().title()})
        slots = {}
        for svc, pat in (("plumber", r"plumb|tap|leak|pipe"), ("electrician", r"electric|wiring|switch"),
                         ("ac_service", r"\bac\b|air.?condition")):
            if re.search(pat, t):
                slots["service"] = svc
                break
        for a in KNOWN_AREAS:
            if a.lower() in t:
                slots["area"] = a
        if "this week" in t:
            slots["date_from"], slots["date_to"] = today.isoformat(), (today + timedelta(days=6 - today.weekday())).isoformat()
        elif "next week" in t:
            mon = today + timedelta(days=7 - today.weekday())
            slots["date_from"], slots["date_to"] = mon.isoformat(), (mon + timedelta(days=6)).isoformat()
        elif "tomorrow" in t:
            slots["date_from"] = slots["date_to"] = (today + timedelta(days=1)).isoformat()
        elif "today" in t:
            slots["date_from"] = slots["date_to"] = today.isoformat()
        else:
            for i, wd in enumerate(WEEKDAYS):
                if wd in t:
                    d = today + timedelta(days=((i - today.weekday()) % 7) or 7)
                    slots["date_from"] = slots["date_to"] = d.isoformat()
                    break
            iso = re.search(r"\d{4}-\d{2}-\d{2}", t)
            if iso:
                slots["date_from"] = slots["date_to"] = iso.group(0)
        if "morning" in t:
            slots["latest_time"] = "12:00"
        elif "afternoon" in t:
            slots["earliest_time"], slots["latest_time"] = "12:00", "17:00"
        elif "evening" in t:
            slots["earliest_time"] = "17:00"
        tm = re.search(r"(not before|after|before)\s+(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?", t)
        if tm:
            hh = int(tm.group(2)) % 12 + (12 if (tm.group(4) or "").startswith("p") else 0) if tm.group(4) else int(tm.group(2))
            hhmm = f"{hh:02d}:{tm.group(3) or '00'}"
            slots["latest_time" if tm.group(1) == "before" else "earliest_time"] = hhmm
        bm = re.search(r"(?:under|below|less than|within|budget(?: of)?|max)\s*(?:₹|rs\.?|inr)?\s*([\d,]+)", t)
        if bm:
            slots["budget"] = int(bm.group(1).replace(",", ""))
        if re.search(r"same .* as last time|same (one|provider|electrician|plumber)", t):
            slots["provider_pref"] = "last"
        bid = re.search(r"\bB-\d+\b", text, re.I)
        if bid:
            slots["target_booking"] = bid.group(0).upper()
        for k in list(slots):
            if k not in SLOT_KEYS:
                slots.pop(k)
        if re.search(VERBS["cancel"], t):
            return Intent("task", "cancel", slots)
        if re.search(VERBS["reschedule"], t) and (ctx.get("action") in (None, "reschedule") or "move" in t or "reschedule" in t):
            return Intent("task", "reschedule", slots)
        if re.search(r"\b(book|find|need|get|want|schedule|hire|arrange|looking|same)\b", t) and ctx.get("state") in (None, "IDLE", "DONE", "FAILED"):
            return Intent("task", "book", slots)
        if slots:
            return Intent("slots", slots=slots)
        if re.search(r"\b(hi|hello|hey|thanks|thank you)\b", t):
            return Intent("smalltalk")
        return Intent("unknown")

    def summarize_reviews(self, reviews):
        out = {}
        for pid, texts in reviews.items():
            out[pid] = f"{len(texts)} review(s): " + (texts[0] if texts else "none")[:90]
        return out
