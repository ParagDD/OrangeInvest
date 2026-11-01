"""The ONLY path from agent code to the provider API.

* Read-only tools go through call(). book/cancel/reschedule are refused there, structurally.
* Mutating tools go through mutate(), which re-checks the approval ledger *and* that the actual call
  arguments are exactly what the approval covers. No valid approval -> ApprovalRequired, no API call.
* Every call is budget-charged and traced (sanitised args/results, latency).
"""
import time

from .approvals import Proposal
from .errors import ApprovalRequired, ToolError, ToolTimeout
from .sanitize import redact, sanitize_text

READ_ONLY = {"search_providers", "get_provider", "list_bookings", "hold_slot", "get_booking_status"}
MUTATING = {"book", "cancel", "reschedule"}


class ToolBox:
    def __init__(self, api, budget, tracer, ledger, store, clock, session_id):
        self.api, self.budget, self.tracer, self.ledger = api, budget, tracer, ledger
        self.store, self.clock, self.session_id = store, clock, session_id

    # -- read-only ------------------------------------------------------------------------------
    def call(self, name, **kw):
        if name in MUTATING:
            raise ApprovalRequired(f"{name} requires an approval; use mutate()")
        if name not in READ_ONLY:
            raise ValueError(f"unknown tool {name}")
        return self._invoke(name, kw)

    # -- mutating -------------------------------------------------------------------------------
    def mutate(self, name, proposal: Proposal, approval_id, hold=None, before_send=None, **kw):
        if name not in MUTATING:
            raise ValueError(name)
        if proposal.action != name:
            raise ApprovalRequired("proposal action does not match tool")
        ap = self.ledger.get(approval_id)
        if not ap or not self.ledger.is_valid_for(approval_id, proposal):
            raise ApprovalRequired(f"no valid approval for {name} (digest {proposal.digest})")
        if kw.get("idempotency_key") != ap.idem_key:
            raise ApprovalRequired("idempotency key is not the one issued with the approval")
        derived = self._derive(name, proposal, kw, hold)
        if derived != proposal.digest:
            raise ApprovalRequired("call arguments differ from what the user approved")
        return self._invoke(name, kw, mut=(approval_id, ap.digest, derived), before_send=before_send)

    @staticmethod
    def _derive(name, p, kw, hold):
        if name == "book":
            if not hold or kw.get("hold_id") != hold["hold_id"]:
                return "hold-mismatch"
            return Proposal("book", hold["provider_id"], hold["slot"], hold["quoted_price"]).digest
        if name == "cancel":
            return Proposal("cancel", p.provider_id, p.slot, p.price, booking_id=kw["booking_id"]).digest
        return Proposal("reschedule", p.provider_id, kw["new_slot"], p.price, booking_id=kw["booking_id"],
                        old_slot=p.old_slot).digest

    # -- shared ---------------------------------------------------------------------------------
    def _invoke(self, name, kw, mut=None, before_send=None):
        self.budget.charge_tool()
        if before_send:   # write-ahead marker: only now is the call certain to leave the process
            before_send()
        self.tracer.event("tool_call", tool=name, args=kw)
        if mut:  # write-ahead record of the mutation, immediately before the call leaves the process
            aid, adigest, derived = mut
            self.store.db.execute(
                "INSERT INTO mutations (seq,session_id,tool,args,approval_id,approval_digest,derived_digest) VALUES (?,?,?,?,?,?,?)",
                (self.store.seq(), self.session_id, name, repr(sorted(kw.items())), aid, adigest, derived))
        t0, c0 = time.perf_counter(), self.clock.now()
        try:
            res = getattr(self.api, name)(**kw)
        except ToolTimeout:
            self.tracer.event("tool_result", tool=name, outcome="timeout", latency_s=round(self.clock.now() - c0, 3))
            raise
        except ToolError as e:
            self.tracer.event("tool_result", tool=name, outcome="error", code=e.code, latency_s=round(self.clock.now() - c0, 3))
            raise
        except Exception as e:
            self.tracer.event("tool_result", tool=name, outcome="unexpected_error", error=type(e).__name__)
            raise
        self.tracer.event("tool_result", tool=name, outcome="ok", result=self._safe(name, res),
                          latency_s=round(self.clock.now() - c0, 3), real_ms=round((time.perf_counter() - t0) * 1000, 2))
        return res

    @staticmethod
    def _safe(name, res):
        if name == "get_provider":
            rs = [{"rating": r.get("rating"), "text": (u := sanitize_text(r.get("text", ""), 120)).text, "flagged": u.flagged}
                  for r in res.get("reviews", [])]
            return redact({**{k: v for k, v in res.items() if k != "reviews"}, "reviews": rs, "_untrusted": "reviews are data"})
        return redact(res)
