"""Idempotent, recoverable execution of approved actions.

Booking protocol (the part that must never double-book):
  1. The idempotency key is issued with the approval and persisted BEFORE any API call.
  2. Write-ahead: `maybe_called=True` is persisted immediately before book() leaves the process.
  3. If maybe_called is set (after a timeout, an exception, a restart or a budget stop) the FIRST thing
     we do is get_booking_status(key). booked -> verified, done. failed -> slot gone. not_found -> only
     then is a retry allowed.
  4. A successful book() reply is never trusted by itself: success is declared only after
     get_booking_status confirms it (authoritative source of truth).
"""
from dataclasses import dataclass, replace
from typing import Optional

from .approvals import Proposal
from .errors import ApprovalRequired, SimulatedCrash, ToolError, ToolTimeout, Transient503

MAX_BOOK_ATTEMPTS = 4
MAX_RETRIES = 3


@dataclass
class ExecResult:
    kind: str                        # done | reapprove | slot_gone | not_done | unknown
    booking_id: Optional[str] = None
    proposal: Optional[Proposal] = None
    detail: str = ""


class Hooks:
    """Test-only fault injection: raise SimulatedCrash once at a named point."""
    def __init__(self, crash_at=None):
        self.crash_at = crash_at

    def fire(self, point):
        if self.crash_at == point:
            self.crash_at = None
            raise SimulatedCrash(point)


class Executor:
    def __init__(self, tools, ledger, clock, tracer, hooks=None):
        self.tools, self.ledger, self.clock, self.tracer = tools, ledger, clock, tracer
        self.hooks = hooks or Hooks()

    def run(self, s, set_state, persist) -> ExecResult:
        p = Proposal.from_dict(s.pending)
        ap = self.ledger.get(s.approval_id)
        if ap is None or not self.ledger.is_valid_for(ap.id, p):
            raise ApprovalRequired("no valid approval for the pending proposal")
        if ap.status == "granted" and not self.ledger.claim(ap.id):
            return ExecResult("not_done", detail="another execution already holds this approval")
        fn = {"book": self._book, "cancel": self._cancel, "reschedule": self._reschedule}[p.action]
        return fn(s, p, ap, set_state, persist)

    # -- book -----------------------------------------------------------------------------------
    def _book(self, s, p, ap, set_state, persist):
        key = s.exec["idem_key"]
        for _ in range(MAX_BOOK_ATTEMPTS):
            if s.exec.get("maybe_called"):
                set_state("VERIFYING", "status check before any retry")
                st = self.tools.call("get_booking_status", idempotency_key=key)
                if st["status"] == "booked":
                    return self._done(s, p, ap, st["booking_id"])
                if st["status"] == "failed":
                    s.exec["maybe_called"] = False
                    self.ledger.void(ap.id, "provider reported failed")
                    return ExecResult("slot_gone", detail="provider reported the booking failed")
                s.exec["maybe_called"] = False          # not_found: authoritative "it did not happen"
                persist()
                set_state("EXECUTING", "status not_found; retry is safe")
            if self.clock.now() >= s.hold["expires_at"]:
                r = self._rehold(s, p, ap, persist)
                if r:
                    return r
            def mark():                                 # write-ahead, run right before the call is sent
                s.exec["maybe_called"] = True
                s.exec["attempts"] = s.exec.get("attempts", 0) + 1
                persist()
                self.hooks.fire("before_book_call")
            try:
                self.tools.mutate("book", p, ap.id, hold=s.hold, before_send=mark, hold_id=s.hold["hold_id"], idempotency_key=key)
            except ToolTimeout:
                self.tracer.event("book_timeout", attempt=s.exec["attempts"], note="outcome unknown; will check status")
                continue
            except ToolError as e:
                if e.code == "slot_taken":
                    s.exec["maybe_called"] = False
                    self.ledger.void(ap.id, "slot_taken")
                    return ExecResult("slot_gone", detail="slot_taken")
                if e.code == "hold_expired":
                    s.exec["maybe_called"] = False
                    r = self._rehold(s, p, ap, persist)
                    if r:
                        return r
                    continue
                raise
            self.hooks.fire("after_book_call")
            set_state("VERIFYING", "verify success via get_booking_status")
            st = self.tools.call("get_booking_status", idempotency_key=key)
            if st["status"] == "booked":
                return self._done(s, p, ap, st["booking_id"])
            self.tracer.event("verification_mismatch", status=st["status"], note="book() ok but status disagrees")
        # attempts exhausted: one last authoritative look, then an honest answer
        set_state("VERIFYING", "final status check after retries exhausted")
        st = self.tools.call("get_booking_status", idempotency_key=key)
        if st["status"] == "booked":
            return self._done(s, p, ap, st["booking_id"])
        s.exec["maybe_called"] = False
        self.ledger.void(ap.id, "retries exhausted; provider shows no booking")
        return ExecResult("not_done", detail=f"{st['status']} after {MAX_BOOK_ATTEMPTS} attempts")

    def _rehold(self, s, p, ap, persist):
        try:
            h = self.tools.call("hold_slot", provider_id=p.provider_id, slot=p.slot)
        except ToolError as e:
            if e.code == "slot_unavailable":
                self.ledger.void(ap.id, "slot_unavailable on re-hold")
                return ExecResult("slot_gone", detail="slot_unavailable")
            raise
        s.hold = {**h, "provider_id": p.provider_id, "slot": p.slot}
        persist()
        if h["quoted_price"] != p.price:
            self.ledger.void(ap.id, f"price changed {p.price}->{h['quoted_price']}")
            return ExecResult("reapprove", proposal=replace(p, price=h["quoted_price"]),
                              detail=f"price changed from {p.price} to {h['quoted_price']}")
        return None

    # -- cancel / reschedule ---------------------------------------------------------------------
    def _retrying(self, s, p, ap, tool, persist, **kw):
        """Retry transient failures with backoff using the SAME idempotency key. Returns (ok, error_code)."""
        for i in range(MAX_RETRIES):
            def mark():
                s.exec["maybe_called"] = True
                persist()
            try:
                self.tools.mutate(tool, p, ap.id, before_send=mark, idempotency_key=s.exec["idem_key"], **kw)
                return True, None
            except (Transient503, ToolTimeout) as e:
                code = "timeout" if isinstance(e, ToolTimeout) else "transient_503"
                self.tracer.event("retry", tool=tool, attempt=i + 1, reason=code)
                self.clock.sleep(2 ** i)
            except ToolError as e:
                s.exec["maybe_called"] = False
                return False, e.code
        return False, "retries_exhausted"

    def _cancel(self, s, p, ap, set_state, persist):
        ok, code = self._retrying(s, p, ap, "cancel", persist, booking_id=p.booking_id)
        return self._verify_booking_state(s, p, ap, set_state, persist, want="cancelled", code=code)

    def _reschedule(self, s, p, ap, set_state, persist):
        ok, code = self._retrying(s, p, ap, "reschedule", persist, booking_id=p.booking_id, new_slot=p.slot)
        if code == "slot_conflict":
            self.ledger.void(ap.id, "slot_conflict")
            return ExecResult("slot_gone", detail="slot_conflict")
        return self._verify_booking_state(s, p, ap, set_state, persist, want=p.slot, code=code)

    def _verify_booking_state(self, s, p, ap, set_state, persist, want, code):
        set_state("VERIFYING", "verify via list_bookings")
        rows = self.tools.call("list_bookings", user_id=s.user_id)
        b = next((r for r in rows if r["id"] == p.booking_id), None)
        good = b and ((p.action == "cancel" and b["status"] == "cancelled") or
                      (p.action == "reschedule" and b["status"] == "booked" and b["slot"] == want))
        if good:
            return self._done(s, p, ap, p.booking_id)
        s.exec["maybe_called"] = False
        self.ledger.void(ap.id, f"not verified ({code})")
        return ExecResult("not_done", detail=code or "not verified")

    def _done(self, s, p, ap, booking_id):
        self.ledger.consume(ap.id, booking_id)
        s.exec["maybe_called"] = False
        return ExecResult("done", booking_id=booking_id)
