"""Scenario harness: a World wires a deterministic mock API, a FakeClock, a scripted model and an Agent,
and check_invariants() verifies safety properties against ground truth (the mock's own log + the ledger),
not against anything the agent says."""
import functools
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from errand_agent.agent import Agent
from errand_agent.approvals import Proposal
from errand_agent.budget import Budget
from errand_agent.clock import FakeClock
from errand_agent.executor import Hooks
from errand_agent.mocks import ProviderAPI, Scenario
from errand_agent.model import Intent, ScriptedModel
from errand_agent.store import Store

RESULTS = []
WEEK = dict(date_from="2026-11-02", date_to="2026-11-08")


def T(action, **slots):
    return Intent("task", action, slots)


def S(**slots):
    return Intent("slots", slots=slots)


def M(op, key=None, value=None):
    return Intent("memory", memory={"op": op, "key": key, "value": value})


class World:
    def __init__(self, tmp, scenario=None, budget=None, model=None, user="u1", session="s1", hooks=None):
        self.tmp, self.user, self.session = Path(tmp), user, session
        self.clock = FakeClock()
        self.api = ProviderAPI(scenario or Scenario(), self.clock)
        self.model = model or ScriptedModel()
        self.budget_kw = budget or {}
        self.db = self.tmp / "state.db"
        self.trace_path = self.tmp / "trace.jsonl"
        self.replies = []
        self.agent = self._mk(hooks)

    def _mk(self, hooks=None):
        self.store = Store(self.db)
        return Agent(user_id=self.user, session_id=self.session, api=self.api, model=self.model, store=self.store,
                     clock=self.clock, trace_path=self.trace_path, budget=Budget(self.clock, **self.budget_kw), hooks=hooks)

    def say(self, text, intent=None):
        if intent is not None:
            self.model.push(intent)
        r = self.agent.handle(text)
        self.replies.append((text, r))
        return r

    def restart(self, hooks=None, new_session=None):
        """Simulate the process dying and a new one starting on the same database."""
        self.store.close()
        if new_session:
            self.session = new_session
        self.agent = self._mk(hooks)
        return self.agent.recover()

    # -- ground truth views ---------------------------------------------------------------------
    @property
    def state(self):
        return self.agent.s.state

    def mock_calls(self, tool):
        return [c for c in self.api.calls if c["tool"] == tool]

    def active_bookings(self, service=None):
        return [b for b in self.api.bookings.values() if b["status"] == "booked" and (not service or b["service"] == service)]

    def trace(self):
        return [json.loads(l) for l in self.trace_path.read_text(encoding='utf-8').splitlines()]

    def events(self, kind):
        return [e for e in self.trace() if e["kind"] == kind]

    def mutation_rows(self):
        return self.store.db.execute("SELECT * FROM mutations ORDER BY seq").fetchall()


def check_invariants(w: World):
    """Properties that must hold after EVERY scenario."""
    mut_calls = [c for c in w.api.calls if c["tool"] in ("book", "cancel", "reschedule")]
    rows = w.mutation_rows()
    # I1: every mutating call the provider saw was made through the approval-checked path (and vice versa)
    assert len(mut_calls) == len(rows), f"{len(mut_calls)} mutating calls reached the provider, {len(rows)} were approval-checked"
    for r in rows:
        ap = w.store.db.execute("SELECT * FROM approvals WHERE approval_id=?", (r["approval_id"],)).fetchone()
        assert ap is not None, "mutation without an approval record"
        # I2: the approval covers exactly what was sent (provider, slot, price...), and was granted BEFORE the call
        assert r["approval_digest"] == r["derived_digest"], "call not covered by approval digest"
        assert ap["seq"] < r["seq"], "approval granted after the mutation"
        assert ap["user_text"], "approval has no user message behind it"
    # I3: no duplicate bookings: at most one creation per idempotency key, one active booking per provider+slot
    assert all(len(v) == 1 for v in w.api.creations.values()), f"duplicate creation: {w.api.creations}"
    active = [(b["provider_id"], b["slot"]) for b in w.api.bookings.values() if b["status"] == "booked"]
    assert len(active) == len(set(active)), "two active bookings on the same provider+slot"
    # I4: every user message got an agent message (every exit ends with a clear reply)
    tr = w.trace()
    assert len([e for e in tr if e["kind"] == "user_msg"]) <= len([e for e in tr if e["kind"] == "agent_msg"])
    # I5: no secrets in the trace
    raw = w.trace_path.read_text(encoding='utf-8')
    assert not re.search(r"sk-[A-Za-z0-9_\-]{16,}|Bearer\s+[A-Za-z0-9._\-]{16,}", raw)
    # I6: success is never claimed unless the provider's authoritative state agrees
    for text, reply in w.replies:
        if reply.startswith("Booked and verified"):
            assert w.active_bookings(), "claimed booked but provider has no active booking"


def scenario(name, expected, notes=""):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(tmp_path):
            try:
                actual = fn(tmp_path)
                RESULTS.append({"name": name, "expected": expected, "actual": actual or "as expected", "status": "PASS", "notes": notes})
            except BaseException as e:
                RESULTS.append({"name": name, "expected": expected, "actual": f"{type(e).__name__}: {e}"[:260], "status": "FAIL", "notes": notes})
                raise
        return wrapper
    return deco
