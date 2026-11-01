> Easiest: use the web console (`python -m errand_agent.webapp`). Pick *Timeout after success*, click *New session*, set the trace filter to *Safety only*, and follow the steps below with the buttons.

# Demo script (3–5 min, two panes: chat left, live trace right)

Right pane: `python -m errand_agent.traceview traces/demo.jsonl --follow`
Left pane:  `python -m errand_agent.cli --session demo --book-script timeout_booked`

1. **Booking + timeout-after-success (≈2 min).**
   `Find a plumber in Baner this week, mornings only, under ₹1,500.` → options appear. Point at the trace:
   `untrusted_flagged` (one review looked like instructions and was withheld).
   `1` → trace shows `hold_slot`, `proposal_shown`. Say: nothing is booked, no approval row exists yet.
   `yes` → trace: `approval_granted` → `book` → `outcome: timeout` → `get_booking_status` → `booked` → `DONE`.
   Say: the book call timed out, but the agent did not retry or claim success; it asked the provider's source of truth first.
   `yes` again → "already done", zero tool calls in the trace.
2. **Prompt injection (≈1 min).** Point at the withheld review in step 1. Then `pytest -q -k injection` and show both injection scenarios
   passing (flagged, and evasive + hijacked model) and the model-proposed `cancel B-2001` being rejected as `ungrounded`.
3. **Price change after approval (≈1 min).** New session: `--price-drift p_pl_1:1200,1350`; pick Ravi Plumbing, wait for the hold to lapse
   (or run `pytest -q -k price_change`), say `yes`; trace shows `approval_voided: price changed 1200->1350` and no `book` call.
4. **Restart (≈30 s).** `pytest -q -k restart` and show `recover` → `get_booking_status` before any `book`.
5. Close on `reports/eval_table.md` (33/33).
