# Design note

**Principle:** the model is a thin, untrusted-ish parser. Code owns state, approvals, idempotency, limits, memory and every sentence that claims something happened. Each must-have is enforced in code and has a test.

## State model
`IDLE → COLLECTING → PRESENTING → AWAITING_APPROVAL → EXECUTING → VERIFYING → DONE | FAILED` (transition table in `agent.py`; an illegal transition raises and is traced). Slots are a typed dict with provenance (`user` | `memory`); required to search: service, area, date range. `_missing()` is a pure function of slots, so the agent can only ask for what is absent. A correction patches one slot and keeps the rest; if a proposal or option list existed, it is voided and the search re-runs. The whole session is persisted to SQLite after every turn and every executor step.

## Approval enforcement
An approval is a **row** in the `approvals` table, created by `Ledger.grant()` only when the user's *raw message* is an exact confirmation word and the digest of the proposal they were last shown equals the pending one. A `Proposal` is `(action, provider, slot, price, booking_id, old_slot)`; its SHA-256 digest binds the approval to those exact facts, so changing provider, slot or price yields a different digest and the old approval simply doesn't cover it. The model has no way to express approval: `Intent` kinds are a closed set, with no "confirm".
Enforcement is structural: `ToolBox.call()` refuses `book/cancel/reschedule`; `ToolBox.mutate()` re-reads the ledger, checks the digest, checks the idempotency key is the one issued with the approval, and re-derives a digest from the *actual call arguments* (hold id/provider/slot/quoted price, booking id, new slot) before the call leaves the process. Price drift is handled by re-checking the hold: if it lapsed and the re-quote differs, the approval is voided and a fresh prompt is shown. `check_invariants()` re-verifies after every scenario, against the mock's own call log, that every provider-side mutation has an approval row granted earlier whose digest matches.

## Idempotency and recovery
The idempotency key is issued with the approval and persisted before any call. Before each `book()` the executor writes `maybe_called=true` (write-ahead, after budget/approval checks). Rule: **if `maybe_called` is set, the next action is always `get_booking_status(key)`**; `booked` → verified, `failed` → slot gone, `not_found` → only then retry with the same key. Even a successful `book()` reply is not believed until `get_booking_status` agrees. `claim()` is an atomic `UPDATE … WHERE status='granted'`, so a repeated "yes" cannot start a second execution; after `DONE`, further confirmations are no-ops with zero tool calls. On restart `recover()` re-enters the executor at the status check (two crash points tested). Attempts are bounded (4 book, 3 cancel/reschedule); every exit, including "unknown outcome", produces a message that never claims success it hasn't verified.

## Untrusted output
Review text is data: control chars stripped, length capped, injection-pattern hits are **withheld**. Anything left goes to a *text-only, quarantined* summarisation call whose output cannot touch state, is stripped of links and "booked/confirmed/cancelled" claims, and is displayed as a labelled, unverified note. `interpret()` never receives tool free-text (tested). Defence in depth against a fully hijacked model: closed intent schema + value validation, and a grounding check requiring cancel/reschedule intents to be supported by the user's own words (a hijacked "cancel B-2001" was rejected in test).

## Memory
SQLite `memory(user, key, value, source, confirmed, forgotten, value_hash)`. Stated facts (area, earliest time, preferred technician) are saved after a successful booking and shown to the user; inferred facts (e.g. same technician ≥2 times in history) are stored **unconfirmed** and never used until the user says so. `view`, `set` (correct), `confirm`, `forget` are chat commands. Forget nulls the value and leaves a hash-only tombstone, so the same inference can't return and the value is never read again.

## Observability
One JSONL trace per session: state transitions, model calls (usage/cost fields; zero for scripted), tool name + args, sanitised results (reviews truncated and flagged), simulated and real latency, approvals, limits, errors. All strings pass through `redact()` (secret-like keys and token patterns). Excerpt of `reports/trace_timeout_after_success.jsonl`:

```
27 approval_granted  A-1  digest 18d6…  {book p_pl_3 2026-11-03T10:30 ₹900}        <- from the user's "yes" on the shown proposal
28 state EXECUTING       "user approved the exact proposal"
29 tool_call   book {hold_id: H-0001, idempotency_key: idem-s1-A-1}
30 tool_result book outcome=timeout                                                 <- booking actually exists, agent can't know
31 book_timeout attempt=1 "outcome unknown; will check status"
32 state VERIFYING      "status check before any retry"                            <- no second book(): status first
33 tool_call   get_booking_status {idem-s1-A-1}
34 tool_result get_booking_status ok {status: booked, booking_id: B-3001}
35 approval_consumed A-1 → B-3001 ;  36 state DONE "verified"
38 agent_msg "Booked and verified with the provider: …"                            <- success claimed only now
```

## What breaks first at 100×, and what I'd change
1. **SQLite single writer + per-call fsync** (autocommit) → move to Postgres; make `claim`/grant transactions and add `SELECT … FOR UPDATE` on the session row. Today an in-process `RLock` serialises a session; across processes only `claim` is atomic.
2. **Synchronous tool calls inside the turn** hold a worker per user, and 3 sequential `get_provider` calls per search add latency → async executor with a durable job queue; fetch details concurrently and cache them.
3. **Recovery only runs when a process picks the session up** → a sweeper that scans sessions stuck in `EXECUTING/VERIFYING` and verifies them, instead of waiting for the user.
4. **Trace files per session** don't aggregate → OpenTelemetry spans to a collector; sampling for `tool_result` bodies.
5. **Injection defence is pattern + structure.** The structure holds; the patterns won't. I'd add a small classifier on untrusted text and measure its false-negative rate on a red-team set.
6. The idempotency story assumes `get_booking_status` stays authoritative and the provider honours idempotency keys on retries; I'd add alerting on `verification_mismatch` events.

## Gaps (honest)
No live-LLM adapter (the `Model` protocol is the seam); no nice-to-have built; demo not recorded (`DEMO.md` is the script); reschedule price/hold semantics are my reading of an ambiguous contract (see README assumptions).
