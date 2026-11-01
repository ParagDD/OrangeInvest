# Three real bugs hit while building

All three were found by the eval suite on a first run, not by reading code. Nothing here is reconstructed after the fact.

## 1. The wall-clock budget counted the user's thinking time
- **What broke:** `price change after approval` failed with "I hit my time limit and stopped" on the user's `yes`.
- **Why:** `Budget.t0` was set once when the agent was created. The scenario has the user wait 11 minutes (so the hold expires) before answering; the 300 s *run* budget was already spent before the agent did any work. A real user pausing to think would have been told the agent "ran out of time".
- **Fix:** `Budget.begin_turn()` resets the clock at the start of every `handle()`/`recover()`. Wall-clock now bounds active work in a turn; model/tool-call caps remain per run.
- **Guard:** `tests/test_evals.py::test_regression_idle_time` (11-minute pause under a 300 s budget must produce no `limit` event). `test_price_change_after_approval` also exercises it.

## 2. A limit hit before any provider call left a live approval and a stuck state
- **What broke:** after the fix above, the new `limit before any provider call` scenario showed the session still in `EXECUTING` with an approval in `executing` status, while the user had just been told "nothing was booked, start again".
- **Why:** `_abort_task` deliberately skipped `EXECUTING`/`VERIFYING` (to keep in-flight bookings resumable) and so also skipped the *not* in-flight case. A later `yes` or `check status` could have resumed an action the user was told was abandoned.
- **Fix:** if no call was sent, `_abort_task` voids every live approval for the session and moves to `FAILED`.
- **Guard:** `test_regression_limit_before_send` asserts state `FAILED`, no approval in `granted`/`executing`, zero `book` calls.

## 3. The write-ahead "maybe sent" marker was set before the budget check
- **What broke:** the same scenario still failed: the reply said "a request may already have been sent, I can't tell you whether it went through" even though the budget refused the call before it left the process.
- **Why:** the executor persisted `maybe_called=True` *then* called `ToolBox.mutate`, whose first step charges the budget. A refusal there left the marker set, so the agent reported uncertainty that did not exist and kept the session in the resumable in-flight state.
- **Fix:** `mutate(..., before_send=...)` runs the write-ahead callback only after the budget charge and the approval checks pass, immediately before the API call (and before the mutation-log row, so a crash at that point leaves both consistent).
- **Guard:** the same `test_regression_limit_before_send` (it asserts the "Nothing was booked" message), plus `test_restart_before_call`, which crashes exactly at the write-ahead point.

## Also caught (not counted as bugs)
- My "evasive" injection review wasn't evasive: the `assistant:` filter pattern caught it, so the hijacked-model scenario never reached the model. Reworded the mock review so it genuinely bypasses the regex.
- The "remember Imran" hint appeared on a plumber search. Scoped it to the same service as the search.

## Sanity check on the tests themselves
I sabotaged the code three ways and confirmed the suite fails each time: (1) blind retry after a timeout (3 scenarios fail), (2) disabling the approval check in `ToolBox.mutate` (the structural-guard scenario fails), (3) trusting `book()` without verification (2 scenarios fail). Mutation 2 is caught only by the guard scenario: the agent's routing independently prevents unapproved calls, so that layer is redundant by design.
