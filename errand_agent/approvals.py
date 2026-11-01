"""Approval ledger. An approval is a ROW that code checks, never something inferred from model output.

A Proposal is the exact action: (action, provider, slot, price, booking id, old slot). Its digest binds an
approval to those exact facts, so changing the provider, slot or price produces a different digest and the
old approval simply does not cover the new action.
"""
import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass(frozen=True)
class Proposal:
    action: str                       # book | cancel | reschedule
    provider_id: str
    slot: str                         # book: the slot; cancel: the booked slot; reschedule: the NEW slot
    price: int
    booking_id: Optional[str] = None  # cancel / reschedule
    old_slot: Optional[str] = None    # reschedule

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return Proposal(**d)


@dataclass
class Approval:
    id: str
    session_id: str
    digest: str
    idem_key: str
    status: str   # granted -> executing -> consumed | voided
    seq: int
    booking_id: Optional[str] = None


class Ledger:
    def __init__(self, store, tracer=None):
        self.s, self.tracer = store, tracer

    def _t(self, kind, **kw):
        if self.tracer:
            self.tracer.event(kind, **kw)

    def _row(self, r):
        return Approval(r["approval_id"], r["session_id"], r["digest"], r["idem_key"], r["status"], r["seq"], r["booking_id"])

    def grant(self, session_id, proposal: Proposal, user_text: str) -> Approval:
        """Called ONLY from the chat layer when the user's raw message is an explicit confirmation of the
        proposal that was last shown to them. One live approval per session."""
        self.s.db.execute("UPDATE approvals SET status='voided', reason='superseded' WHERE session_id=? AND status='granted'",
                          (session_id,))
        seq = self.s.seq()
        aid = f"A-{seq}"
        idem = f"idem-{session_id}-{aid}"
        self.s.db.execute(
            "INSERT INTO approvals (approval_id,session_id,digest,proposal,idem_key,status,seq,user_text) VALUES (?,?,?,?,?,'granted',?,?)",
            (aid, session_id, proposal.digest, json.dumps(proposal.to_dict()), idem, seq, user_text[:200]))
        self._t("approval_granted", approval_id=aid, digest=proposal.digest, proposal=proposal.to_dict())
        return self.get(aid)

    def get(self, aid) -> Optional[Approval]:
        r = self.s.db.execute("SELECT * FROM approvals WHERE approval_id=?", (aid,)).fetchone()
        return self._row(r) if r else None

    def is_valid_for(self, aid, proposal: Proposal) -> bool:
        r = self.s.db.execute("SELECT digest,status FROM approvals WHERE approval_id=?", (aid,)).fetchone()
        return bool(r) and r["digest"] == proposal.digest and r["status"] in ("granted", "executing")

    def claim(self, aid) -> bool:
        """Atomic granted -> executing. Exactly one caller wins; a repeated 'yes' cannot start a 2nd execution."""
        cur = self.s.db.execute("UPDATE approvals SET status='executing' WHERE approval_id=? AND status='granted'", (aid,))
        return cur.rowcount == 1

    def consume(self, aid, booking_id=None):
        self.s.db.execute("UPDATE approvals SET status='consumed', booking_id=? WHERE approval_id=?", (booking_id, aid))
        self._t("approval_consumed", approval_id=aid, booking_id=booking_id)

    def void(self, aid, reason):
        self.s.db.execute("UPDATE approvals SET status='voided', reason=? WHERE approval_id=? AND status IN ('granted','executing')",
                          (reason, aid))
        self._t("approval_voided", approval_id=aid, reason=reason)

    def void_session(self, session_id, reason):
        for r in self.s.db.execute("SELECT approval_id FROM approvals WHERE session_id=? AND status IN ('granted','executing')",
                                   (session_id,)).fetchall():
            self.void(r["approval_id"], reason)
