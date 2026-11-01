"""The agent: an explicit state machine around a thin model interface.

Division of labour:
  model  -> turns language into a validated Intent; writes quarantined, labelled review notes.
  code   -> owns state, slots, approvals, idempotency, retries, budgets, memory, traces, and every
            sentence that claims something happened.
"""
import json
import re
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

from .approvals import Ledger, Proposal
from .budget import Budget
from .errors import ApprovalRequired, BudgetExceeded, ToolError
from .executor import Executor
from .memory import Memory
from .model import MEMORY_KEYS, grounded, validate_intent
from .sanitize import neutralize_note, sanitize_text
from .toolbox import ToolBox
from .trace import Tracer

ALLOWED = {
    "IDLE": {"COLLECTING", "PRESENTING", "AWAITING_APPROVAL", "FAILED"},
    "COLLECTING": {"COLLECTING", "PRESENTING", "AWAITING_APPROVAL", "IDLE", "FAILED"},
    "PRESENTING": {"COLLECTING", "AWAITING_APPROVAL", "FAILED", "IDLE"},
    "AWAITING_APPROVAL": {"EXECUTING", "PRESENTING", "COLLECTING", "FAILED", "IDLE"},
    "EXECUTING": {"VERIFYING", "AWAITING_APPROVAL", "PRESENTING", "COLLECTING", "DONE", "FAILED"},
    "VERIFYING": {"EXECUTING", "DONE", "FAILED", "AWAITING_APPROVAL", "PRESENTING", "COLLECTING"},
    "DONE": {"IDLE", "COLLECTING", "PRESENTING", "AWAITING_APPROVAL"},
    "FAILED": {"IDLE", "COLLECTING", "PRESENTING", "AWAITING_APPROVAL", "EXECUTING", "VERIFYING", "DONE"},
}
CONFIRM = {"yes", "y", "yep", "yeah", "yup", "confirm", "confirmed", "ok", "okay", "sure", "go ahead", "do it",
           "book it", "yes please", "yes confirm", "please do", "confirm it"}
DECLINE = {"no", "n", "nope", "cancel that", "never mind", "nevermind", "stop", "dont", "dont do it", "no thanks", "no thank you"}
STATUS = {"check status", "status", "check", "check booking status"}
MEM_TO_SLOT = {"area": "area", "earliest_time": "earliest_time", "preferred_technician": "technician_pref"}
LIMIT_WORDS = {"tool_calls": "tool-call limit", "model_calls": "model-turn limit", "wall_clock": "time limit"}
SERVICE_NAMES = {"plumber": "plumber", "electrician": "electrician", "ac_service": "AC servicing"}


def money(n):
    return f"₹{int(n):,}"


def fmt_slot(slot):
    d = datetime.strptime(slot, "%Y-%m-%dT%H:%M")
    return f"{d:%a} {d.day} {d:%b}, {d:%H:%M}"


def norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", "", text.lower().replace("'", ""))).strip()


@dataclass
class Session:
    session_id: str
    user_id: str
    state: str = "IDLE"
    action: Optional[str] = None
    slots: dict = field(default_factory=dict)       # key -> {"value":..., "source": "user"|"memory"}
    options: list = field(default_factory=list)
    hold: Optional[dict] = None
    pending: Optional[dict] = None                  # the Proposal awaiting approval
    pending_meta: dict = field(default_factory=dict)
    shown_digest: Optional[str] = None              # digest of the proposal the user was last shown
    approval_id: Optional[str] = None
    exec: dict = field(default_factory=dict)        # idem_key, maybe_called, attempts
    target: Optional[dict] = None
    history: Optional[list] = None
    review_notes: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)
    result: Optional[dict] = None
    turns: int = 0


class Agent:
    def __init__(self, *, user_id, session_id, api, model, store, clock, trace_path, budget=None, hooks=None):
        self.user_id, self.session_id, self.model, self.store, self.clock = user_id, session_id, model, store, clock
        self.trace = Tracer(trace_path, clock, session_id)
        self.memory = Memory(store, self.trace)
        self.ledger = Ledger(store, self.trace)
        self.budget = budget or Budget(clock)
        self.tools = ToolBox(api, self.budget, self.trace, self.ledger, store, clock, session_id)
        self.executor = Executor(self.tools, self.ledger, clock, self.trace, hooks)
        self.lock = threading.RLock()
        self.s = self._load() or Session(session_id, user_id)

    # ------------------------------------------------------------------ persistence / state ------
    def _load(self):
        r = self.store.db.execute("SELECT data FROM sessions WHERE session_id=?", (self.session_id,)).fetchone()
        return Session(**json.loads(r["data"])) if r else None

    def _save(self):
        self.store.db.execute("INSERT INTO sessions (session_id,data) VALUES (?,?) ON CONFLICT(session_id) DO UPDATE SET data=excluded.data",
                              (self.session_id, json.dumps(asdict(self.s))))

    def _to(self, new, why=""):
        old = self.s.state
        if new == old:
            return
        if new not in ALLOWED[old]:
            raise RuntimeError(f"illegal transition {old} -> {new}")
        self.trace.event("state", frm=old, to=new, why=why)
        self.s.state = new

    def _exec_state(self, new, why):
        self._to(new, why)
        self._save()

    def _val(self, k):
        return self.s.slots.get(k, {}).get("value")

    # ------------------------------------------------------------------ entry points -------------
    def handle(self, text: str) -> str:
        with self.lock:
            self.s.turns += 1
            self.trace.turn = self.s.turns
            self.budget.begin_turn()
            self.trace.event("user_msg", text=text)
            reply = self._guarded(lambda: self._turn(text))
            self._save()
            self.trace.event("agent_msg", text=reply, state=self.s.state)
            return reply

    def recover(self) -> Optional[str]:
        """Called after a restart. If a mutation was in flight, verify with the provider BEFORE anything else."""
        with self.lock:
            if self.s.state in ("EXECUTING", "VERIFYING") and self.s.exec.get("idem_key"):
                self.trace.turn = self.s.turns
                self.budget.begin_turn()
                self.trace.event("recover", state=self.s.state, maybe_called=self.s.exec.get("maybe_called"))
                reply = self._guarded(self._execute)
                self._save()
                self.trace.event("agent_msg", text=reply, state=self.s.state)
                return reply
            return None

    def _guarded(self, fn):
        try:
            return fn()
        except BudgetExceeded as e:
            return self._on_limit(e.reason)
        except ApprovalRequired as e:
            self.trace.event("error", type="ApprovalRequired", msg=str(e))
            self._abort_task()
            return ("I stopped because I could not verify your approval for that step. Nothing was booked, "
                    "cancelled or changed. Please start again.")
        except Exception as e:  # unknown errors always end with a clear message
            self.trace.event("error", type=type(e).__name__, msg=str(e))
            return self._on_error(e)

    # ------------------------------------------------------------------ failure exits ------------
    def _in_flight(self):
        return bool(self.s.exec.get("maybe_called")) and self.s.state in ("EXECUTING", "VERIFYING")

    def _abort_task(self):
        s = self.s
        if s.state in ("EXECUTING", "VERIFYING"):      # nothing was sent: do not leave a live approval behind
            self.ledger.void_session(s.session_id, "aborted before any provider call")
            self._to("FAILED", "aborted")
        elif s.state not in ("DONE", "FAILED"):
            self._to("FAILED", "aborted")
        s.pending, s.shown_digest = None, None

    def _unknown_msg(self, why):
        a = (self.s.pending or {}).get("action", "request")
        return (f"{why} A {a} request may already have been sent, so I can't tell you whether it went through, "
                f"and I will not retry blindly. Ask me to 'check status' (in a new session if needed) and I'll verify "
                f"with the provider before doing anything else. Reference: {self.s.exec.get('idem_key')}")

    def _on_limit(self, reason):
        self.trace.event("limit", reason=reason, model_calls=self.budget.model_calls, tool_calls=self.budget.tool_calls)
        word = LIMIT_WORDS.get(reason, reason)
        if self._in_flight():
            return self._unknown_msg(f"I hit my {word} before I could confirm the outcome.")
        self._abort_task()
        return f"I hit my {word} and stopped. Nothing was booked, cancelled or changed. You can start again with a new request."

    def _on_error(self, e):
        if self._in_flight():
            return self._unknown_msg("Something unexpected went wrong while finishing this.")
        self._abort_task()
        return (f"Something unexpected went wrong on my side ({type(e).__name__}). "
                "Nothing was booked, cancelled or changed. Please try again.")

    # ------------------------------------------------------------------ turn routing -------------
    def _turn(self, text):
        s, n = self.s, norm(text)
        if s.state in ("EXECUTING", "VERIFYING"):
            if n in STATUS or n in CONFIRM:
                return self._execute()
            return "I'm still confirming your last request with the provider. Say 'check status' and I'll verify it."
        if n in CONFIRM:
            return self._on_confirm(text)
        if n in STATUS:
            return self._help_text("There's no request in progress.")
        if n in DECLINE:
            return self._on_decline()
        m = re.fullmatch(r"(?:option )?(\d+)", n)
        if m:
            return self._on_pick(int(m.group(1)))
        return self._apply_intent(self._interpret(text), text)

    def _interpret(self, text):
        s = self.s
        ctx = {"today": self.clock.today_iso(), "state": s.state, "action": s.action,
               "known": {k: v["value"] for k, v in s.slots.items()}, "n_options": len(s.options)}
        self.budget.charge_model()
        raw = self.model.interpret(text, ctx)
        intent, why = validate_intent(raw)
        if intent and not grounded(intent, text):
            intent, why = None, "ungrounded: action not supported by the user's own words"
        self.trace.event("model_call", fn="interpret", usage=getattr(self.model, "last_usage", None),
                         intent=asdict(raw) if hasattr(raw, "__dataclass_fields__") else str(raw),
                         accepted=intent is not None, rejected_reason=why)
        return intent

    def _apply_intent(self, intent, text):
        s = self.s
        if intent is None:
            return self._help_text("I didn't understand that as a request I can act on.")
        if intent.kind == "memory":
            return self._do_memory(intent.memory)
        if intent.kind == "decline":
            return self._on_decline()
        if intent.kind in ("smalltalk", "unknown"):
            return self._help_text("")
        if intent.kind == "task":
            if s.state in ("IDLE", "DONE", "FAILED") or s.action != intent.action:
                self._new_task(intent.action)
            self._apply_slots(intent.slots, intent.clear)
            return self._advance()
        if not s.action:
            return self._help_text("")
        self._apply_slots(intent.slots, intent.clear)
        return self._advance()

    # ------------------------------------------------------------------ slots --------------------
    def _new_task(self, action):
        s = self.s
        self.ledger.void_session(s.session_id, "new task")
        s.action, s.slots, s.options, s.hold, s.pending, s.pending_meta = action, {}, [], None, None, {}
        s.shown_digest, s.approval_id, s.exec, s.target, s.history, s.result = None, None, {}, None, None, None
        s.review_notes, s.flags = {}, []
        self._to("COLLECTING", f"new {action} task")
        usable = self.memory.usable(self.user_id)
        for mk, sk in MEM_TO_SLOT.items():
            if mk in usable:
                s.slots[sk] = {"value": usable[mk], "source": "memory"}
        if usable:
            self.trace.event("memory_applied", keys=sorted(usable))

    def _apply_slots(self, slots, clear):
        s, changed = self.s, []
        for k in clear:
            if s.slots.pop(k, None) is not None:
                changed.append(k)
        for k, v in slots.items():
            old = s.slots.get(k)
            if not old or old["value"] != v:
                changed.append(k)
            s.slots[k] = {"value": v, "source": "user"}
        if "date_from" in slots and "date_to" not in slots:
            s.slots["date_to"] = {"value": slots["date_from"], "source": "user"}
        if changed:
            self.trace.event("slots_changed", changed=changed, kept=sorted(k for k in s.slots if k not in changed))
            if "service" in changed:
                s.history, s.target = None, None
            if s.pending or s.options:
                self._drop_pending(f"details changed ({', '.join(changed)})")
        return changed

    def _drop_pending(self, why):
        s = self.s
        self.ledger.void_session(s.session_id, why)
        s.pending, s.pending_meta, s.shown_digest, s.hold, s.approval_id, s.exec, s.options = None, {}, None, None, None, {}, []
        if s.state in ("PRESENTING", "AWAITING_APPROVAL"):
            self._to("COLLECTING", why)

    def _missing(self):
        s, v = self.s, self._val
        if s.action == "book":
            return [n for n, k in (("service", "service"), ("area", "area"), ("date range", "date_from")) if not v(k)]
        if s.action == "cancel":
            return [] if (v("target_booking") or v("service")) else ["which booking (service or booking id)"]
        if s.action == "reschedule":
            m = [] if (v("target_booking") or v("service")) else ["which booking (service or booking id)"]
            return m + ([] if v("date_from") else ["the new date"])
        return ["what you'd like to do"]

    def _known_text(self):
        s = self.s
        parts = []
        for k, label in (("service", "service"), ("area", "area"), ("date_from", "from"), ("date_to", "to"),
                         ("earliest_time", "not before"), ("latest_time", "before"), ("budget", "budget")):
            v = self._val(k)
            if v is not None:
                src = " (saved preference)" if s.slots[k]["source"] == "memory" else ""
                parts.append(f"{label}: {money(v) if k == 'budget' else v}{src}")
        return ", ".join(parts)

    def _advance(self):
        s = self.s
        missing = self._missing()
        if missing:
            self._to("COLLECTING", f"missing {missing}")
            known = self._known_text()
            return (f"Got it. " + (f"So far: {known}. " if known else "") + "I still need: " + ", ".join(missing) + ".")
        if s.action == "book":
            return self._search_and_present()
        return self._resolve_target()

    # ------------------------------------------------------------------ search / present ---------
    def _history(self, force=False):
        s = self.s
        if s.history is None or force:
            s.history = self.tools.call("list_bookings", user_id=self.user_id)
            self._infer_prefs(s.history)
        return s.history

    def _infer_prefs(self, hist):
        from collections import Counter
        past = [b for b in hist if b["status"] == "completed" and b.get("technician")]
        if past:
            tech, n = Counter(b["technician"] for b in past).most_common(1)[0]
            if n >= 2:
                self.memory.propose_inferred(self.user_id, "preferred_technician", tech)

    def _search_and_present(self):
        s, v = self.s, self._val
        hist, notes, pref_provider = self._history(), [], None
        if v("provider_pref") == "last":
            past = sorted([b for b in hist if b["service"] == v("service") and b["status"] == "completed"],
                          key=lambda b: b["slot"], reverse=True)
            if past:
                pref_provider = past[0]["provider_id"]
                notes.append(f"Using your last {SERVICE_NAMES[v('service')]} provider: {past[0]['provider_name']} "
                             f"(technician {past[0]['technician']}).")
            else:
                notes.append("I couldn't find a previous booking for that service, so I'm showing all options.")
        lo = v("date_from")
        hi = v("date_to") or lo
        res = self.tools.call("search_providers", service=v("service"), area=v("area"), date_range=[lo, hi])
        e, l, b = v("earliest_time"), v("latest_time"), v("budget")
        opts, st = [], {"slots": 0, "time_ok": 0}
        for p in res:
            for slot in p["free_slots"]:
                st["slots"] += 1
                hhmm = slot[11:16]
                if (e and hhmm < e) or (l and hhmm >= l):
                    continue
                st["time_ok"] += 1
                if b and p["price"] > b:
                    continue
                opts.append({"provider_id": p["id"], "name": p["name"], "slot": slot, "price": p["price"],
                             "rating": p["rating"], "technician": p["attributes"].get("technician")})
        if not opts:
            self._to("COLLECTING", "no options")
            why = (f"no {SERVICE_NAMES[v('service')]} has free slots in {v('area')} between {lo} and {hi}" if not res else
                   "none of the free slots fall inside your time window" if not st["time_ok"] else
                   f"everything inside your time window is above your {money(b)} budget")
            return f"I couldn't find a match: {why}. Tell me what to relax (dates, time window or budget) and I'll search again."
        tpref = v("technician_pref")
        if pref_provider:
            keep = [o for o in opts if o["provider_id"] == pref_provider]
            if keep:
                opts = keep
            else:
                notes.append("Your usual provider has no free slot that fits, so these are alternatives.")
        opts.sort(key=lambda o: (0 if o["provider_id"] == pref_provider else 1, 0 if (tpref and o["technician"] == tpref) else 1,
                                 o["price"], -o["rating"], o["slot"]))
        per, top = {}, []
        for o in opts:
            if per.get(o["provider_id"], 0) < 2 and len(top) < 5:
                per[o["provider_id"]] = per.get(o["provider_id"], 0) + 1
                top.append(o)
        s.options = top
        self._review_notes(top)
        self._to("PRESENTING", "options found")
        return self._options_text(notes)

    def _review_notes(self, top):
        """Reviews are untrusted DATA. Flagged ones are withheld; the rest go to a text-only, quarantined call."""
        s = self.s
        pids = list(dict.fromkeys(o["provider_id"] for o in top))[:3]
        safe, withheld = {}, 0
        for pid in pids:
            prov = self.tools.call("get_provider", provider_id=pid)
            us = [sanitize_text(r.get("text", ""), 200) for r in prov.get("reviews", [])]
            withheld += sum(u.flagged for u in us)
            safe[pid] = [u.text for u in us if not u.flagged]
        if withheld:
            self.trace.event("untrusted_flagged", withheld=withheld, note="review text looked like instructions; withheld")
            s.flags.append(f"{withheld} review(s) were withheld because they looked like instructions rather than feedback.")
        if any(safe.values()):
            self.budget.charge_model()
            raw = self.model.summarize_reviews({k: v for k, v in safe.items() if v})
            self.trace.event("model_call", fn="summarize_reviews", usage=getattr(self.model, "last_usage", None), providers=pids)
            for pid, note in (raw or {}).items():
                if pid in safe:
                    s.review_notes[pid] = neutralize_note(note)

    def _options_text(self, notes=()):
        s = self.s
        lines = list(notes)
        lines.append("Here's what I found (prices are from search; the final price is confirmed before booking):")
        for i, o in enumerate(s.options, 1):
            lines.append(f"  {i}. {o['name']} · {fmt_slot(o['slot'])} · {money(o['price'])} · ★{o['rating']} · technician {o['technician']}")
        shown = {o["provider_id"] for o in s.options}
        nl = [f"   - {pid}: {n}" for pid, n in s.review_notes.items() if pid in shown]
        if nl:
            lines.append("Review notes (auto-summary of customer reviews, unverified):")
            lines += nl
        lines += [f"Note: {f}" for f in s.flags]
        pend = self.memory.pending_inferred(self.user_id).get("preferred_technician")
        relevant = pend and any(b.get("technician") == pend and b["service"] == self._val("service") for b in (s.history or []))
        if relevant and "hint_shown" not in s.flags:
            s.flags.append("hint_shown")
            lines.append(f"I noticed you've used {pend} before. Say 'remember {pend}' if you want me to prefer them; "
                         "I won't use that unless you confirm.")
        lines.append("Reply with a number to pick one.")
        return "\n".join(lines)

    def _re_present(self):
        s = self.s
        if s.options:
            self._to("PRESENTING", "remaining options")
            return self._options_text()
        self._to("COLLECTING", "no options left")
        return "There are no other options left for those details. Tell me a different date, time window or budget."

    # ------------------------------------------------------------------ cancel / reschedule ------
    def _resolve_target(self):
        s, v = self.s, self._val
        rows = self._history(force=True)
        active = [b for b in rows if b["status"] == "booked"]
        cands = [b for b in active if (b["id"] == v("target_booking") if v("target_booking") else b["service"] == v("service"))]
        if not cands:
            self._to("FAILED", "no matching booking")
            return "I couldn't find an upcoming booking matching that, so nothing was changed."
        if len(cands) > 1:
            self._to("COLLECTING", "ambiguous booking")
            return ("You have more than one matching booking. Which one (give me the booking id)?\n" +
                    "\n".join(f"  {b['id']}: {b['provider_name']} · {fmt_slot(b['slot'])}" for b in cands))
        b = s.target = cands[0]
        meta = {"provider_name": b["provider_name"], "service": b["service"], "area": b.get("area", "")}
        if s.action == "cancel":
            return self._set_pending(Proposal("cancel", b["provider_id"], b["slot"], b["price"], booking_id=b["id"]), meta)
        lo, hi = v("date_from"), v("date_to") or v("date_from")
        res = self.tools.call("search_providers", service=b["service"], area=b.get("area") or v("area") or "Baner", date_range=[lo, hi])
        e, l = v("earliest_time"), v("latest_time")
        slots = [sl for p in res if p["id"] == b["provider_id"] for sl in p["free_slots"]
                 if not ((e and sl[11:16] < e) or (l and sl[11:16] >= l))]
        if not slots:
            self._to("COLLECTING", "no slots at provider")
            return f"{b['provider_name']} has no free slots between {lo} and {hi}. Give me another date or time window."
        s.options = [{"provider_id": b["provider_id"], "name": b["provider_name"], "slot": sl, "price": b["price"],
                      "rating": next(p["rating"] for p in res if p["id"] == b["provider_id"]), "technician": b["technician"]}
                     for sl in sorted(slots)[:5]]
        s.pending_meta = meta
        self._to("PRESENTING", "reschedule options")
        return (f"Your {SERVICE_NAMES[b['service']]} booking {b['id']} with {b['provider_name']} is on {fmt_slot(b['slot'])}. "
                "Available new slots:\n" + "\n".join(f"  {i}. {fmt_slot(o['slot'])} · {money(o['price'])}" for i, o in enumerate(s.options, 1))
                + "\nReply with a number to pick one.")

    # ------------------------------------------------------------------ choose / approval --------
    def _on_pick(self, n):
        s = self.s
        if s.state not in ("PRESENTING", "AWAITING_APPROVAL") or not s.options:
            return self._help_text("There's nothing to pick from right now.")
        if not 1 <= n <= len(s.options):
            return f"Please choose a number between 1 and {len(s.options)}."
        opt = s.options[n - 1]
        self.ledger.void_session(s.session_id, "user chose a different option")
        meta = {"provider_name": opt["name"], "service": self._val("service") or (s.target or {}).get("service", ""),
                "area": self._val("area") or (s.target or {}).get("area", "")}
        if s.action == "book":
            try:
                h = self.tools.call("hold_slot", provider_id=opt["provider_id"], slot=opt["slot"])
            except ToolError as e:
                if e.code != "slot_unavailable":
                    raise
                s.options = [o for o in s.options if o is not opt]
                return "Sorry, that slot was just taken. " + self._re_present()
            s.hold = {**h, "provider_id": opt["provider_id"], "slot": opt["slot"]}
            note = ""
            if h["quoted_price"] != opt["price"]:
                note = f"Note: the quoted price is {money(h['quoted_price'])} (search showed {money(opt['price'])}).\n"
            b = self._val("budget")
            if b and h["quoted_price"] > b:
                note += f"Note: this is above your {money(b)} budget.\n"
            return note + self._set_pending(Proposal("book", opt["provider_id"], opt["slot"], h["quoted_price"]), meta)
        t = s.target
        return self._set_pending(Proposal("reschedule", opt["provider_id"], opt["slot"], opt["price"],
                                          booking_id=t["id"], old_slot=t["slot"]), meta)

    def _set_pending(self, p: Proposal, meta):
        s = self.s
        s.pending, s.pending_meta, s.approval_id, s.exec = p.to_dict(), meta, None, {}
        self._to("AWAITING_APPROVAL", f"{p.action} proposed")
        return self._confirm_prompt()

    def _confirm_prompt(self):
        s = self.s
        p, m = Proposal.from_dict(s.pending), s.pending_meta
        if p.action == "book":
            line = f"BOOK {m['provider_name']} ({p.provider_id}) · {fmt_slot(p.slot)} · {money(p.price)} · {SERVICE_NAMES.get(m['service'], m['service'])} in {m['area']}"
            extra = f"\nThe slot is held until {self.clock.hhmm(s.hold['expires_at'])}." if s.hold else ""
        elif p.action == "cancel":
            line = f"CANCEL booking {p.booking_id} with {m['provider_name']} ({p.provider_id}) · {fmt_slot(p.slot)} · {money(p.price)}"
            extra = ""
        else:
            line = (f"RESCHEDULE booking {p.booking_id} with {m['provider_name']} ({p.provider_id}) from "
                    f"{fmt_slot(p.old_slot)} to {fmt_slot(p.slot)} · {money(p.price)}")
            extra = ""
        s.shown_digest = p.digest
        self.trace.event("proposal_shown", digest=p.digest, proposal=p.to_dict())
        return f"Please confirm this exact action:\n  {line}{extra}\nReply 'yes' to confirm or 'no' to drop it."

    def _on_confirm(self, text):
        s = self.s
        if s.state == "AWAITING_APPROVAL" and s.pending:
            p = Proposal.from_dict(s.pending)
            if s.shown_digest != p.digest:
                self.trace.event("approval_refused", reason="proposal changed since it was shown")
                return "The details changed since I last showed them, so I need a fresh confirmation.\n" + self._confirm_prompt()
            ap = self.ledger.grant(s.session_id, p, text)
            s.approval_id = ap.id
            s.exec = {"idem_key": ap.idem_key, "maybe_called": False, "attempts": 0}
            self._exec_state("EXECUTING", "user approved the exact proposal")
            return self._execute()
        if s.state == "DONE" and s.result:
            self.trace.event("duplicate_confirmation_ignored")
            return f"That's already done ({self._done_line()}). I haven't taken any further action."
        return "There's nothing waiting for your confirmation right now."

    def _on_decline(self):
        s = self.s
        if s.pending:
            self._drop_pending("user declined")
            return "Okay, I've dropped that. Nothing was booked, cancelled or changed. Tell me what you'd like instead."
        return self._help_text("There's nothing to decline right now.")

    # ------------------------------------------------------------------ execution ----------------
    def _execute(self):
        res = self.executor.run(self.s, self._exec_state, self._save)
        return self._after_exec(res)

    def _done_line(self):
        r = self.s.result
        return f"{r['action']} {r['booking_id']} with {r['provider_name']}, {fmt_slot(r['slot'])}, {money(r['price'])}"

    def _after_exec(self, res):
        s = self.s
        p = Proposal.from_dict(s.pending)
        m = s.pending_meta
        if res.kind == "done":
            s.result = {"action": p.action, "booking_id": res.booking_id, "provider_name": m.get("provider_name"),
                        "slot": p.slot, "price": p.price}
            self._to("DONE", "verified")
            verb = {"book": "Booked", "cancel": "Cancelled", "reschedule": "Rescheduled"}[p.action]
            return f"{verb} and verified with the provider: {self._done_line()}." + self._remember_after_success(p)
        if res.kind == "reapprove":
            s.pending, s.shown_digest, s.approval_id, s.exec = res.proposal.to_dict(), None, None, {}
            self._to("AWAITING_APPROVAL", "price changed; fresh approval required")
            return (f"Heads-up: {res.detail}. Your earlier 'yes' no longer applies and I have not booked anything.\n"
                    + self._confirm_prompt())
        if res.kind == "slot_gone":
            s.options = [o for o in s.options if not (o["provider_id"] == p.provider_id and o["slot"] == p.slot)]
            s.pending, s.shown_digest, s.hold, s.approval_id, s.exec = None, None, None, None, {}
            self._to("PRESENTING" if s.options else "COLLECTING", "slot gone")
            return (f"That slot is no longer available ({res.detail}). I verified nothing was booked or changed. "
                    + self._re_present_text())
        s.pending, s.shown_digest, s.approval_id = None, None, None
        self._to("FAILED", "not done")
        what = {"book": "booking", "cancel": "cancellation", "reschedule": "reschedule"}[p.action]
        tail = {"book": "The provider's records show no booking, so nothing was charged or reserved.",
                "cancel": "The booking is still active.", "reschedule": "The booking is still at its original time."}[p.action]
        return f"I couldn't complete the {what} ({res.detail}). {tail} You can ask me to try again."

    def _re_present_text(self):
        s = self.s
        if s.options:
            return "Remaining options:\n" + "\n".join(
                f"  {i}. {o['name']} · {fmt_slot(o['slot'])} · {money(o['price'])}" for i, o in enumerate(s.options, 1)) + "\nReply with a number to pick one."
        return "There are no other options left for those details. Tell me a different date, time window or budget."

    def _remember_after_success(self, p):
        if p.action != "book":
            return ""
        saved = []
        for mk, sk in MEM_TO_SLOT.items():
            sv = self.s.slots.get(sk)
            if sv and sv["source"] == "user":
                self.memory.set_stated(self.user_id, mk, sv["value"])
                saved.append(f"{mk}={sv['value']}")
        return (f"\nSaved to your preferences (stated by you): {', '.join(saved)}. Say 'what do you remember' to view or "
                "'forget my area' to remove." if saved else "")

    # ------------------------------------------------------------------ memory -------------------
    def _do_memory(self, m):
        s, op, key = self.s, m["op"], m.get("key")
        if op == "view":
            facts = self.memory.view(self.user_id)
            if not facts:
                return "I haven't saved any preferences about you."
            return "Here's what I have saved:\n" + "\n".join(
                f"  - {f['key']}: {f['value']} ({'stated by you' if f['source'] == 'stated' else 'inferred by me, ' + ('confirmed' if f['confirmed'] else 'unconfirmed, NOT used')})"
                for f in facts)
        if op == "set":
            self.memory.set_stated(self.user_id, key, m["value"])
            msg = f"Saved: {key} = {m['value']} (stated by you)."
            if s.action and MEM_TO_SLOT[key] in ("area", "earliest_time"):
                self._apply_slots({MEM_TO_SLOT[key]: m["value"]}, [])
                if s.state in ("COLLECTING", "PRESENTING"):
                    return msg + "\n" + self._advance()
            return msg
        if op in ("forget", "reject"):
            ok = self.memory.forget(self.user_id, key)
            if s.action and s.slots.pop(MEM_TO_SLOT[key], None) is not None:
                self._drop_pending(f"forgot {key}") if (s.pending or s.options) else None
            return (f"Forgotten: {key}. I won't use it again." if ok else f"I don't have anything saved for {key}.") + (
                "\n" + self._advance() if s.action and s.state == "COLLECTING" and self._missing() else "")
        if op == "confirm":
            pend = self.memory.pending_inferred(self.user_id)
            if key in pend and self.memory.confirm(self.user_id, key):
                if s.action:
                    s.slots[MEM_TO_SLOT[key]] = {"value": pend[key], "source": "memory"}
                return f"Okay, I'll prefer {pend[key]} from now on. Say 'forget my technician' to undo that."
            usable = self.memory.usable(self.user_id)
            if key in usable:
                return f"I already have {usable[key]} saved as your {key.replace('_', ' ')}. Say 'forget my {key.replace('_', ' ')}' first if you want to change it."
            return "I don't have an unconfirmed guess like that to confirm."
        return self._help_text("")

    def _help_text(self, lead):
        s = self.s
        base = ("I can book, reschedule or cancel home-service errands (plumber, electrician, AC servicing). "
                "For example: 'Find a plumber in Baner this week, mornings only, under ₹1,500.'")
        if s.state == "AWAITING_APPROVAL" and s.pending:
            return (lead + " " if lead else "") + "I'm waiting for your yes/no on this:\n" + self._confirm_prompt()
        return (lead + " " if lead else "") + base
