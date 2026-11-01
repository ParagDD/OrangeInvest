"""Interactive chat (no LLM needed: a rule-based model parses the demo phrasings).
   python -m errand_agent.cli [--user u1] [--session NAME] [--book-script timeout_booked,success] [--seed 7]
Trace for the session is written to traces/<session>.jsonl; watch it live with
   python -m errand_agent.traceview traces/<session>.jsonl --follow
"""
import argparse
import time

from .agent import Agent
from .clock import Clock, DEMO_START
from .mocks import ProviderAPI, Scenario
from .model import RuleModel
from .store import Store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="u1")
    ap.add_argument("--session", default=f"demo-{int(time.time())}")
    ap.add_argument("--db", default="state.db")
    ap.add_argument("--book-script", default="", help="comma list of success|slot_taken|timeout_booked|timeout_not_booked")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--price-drift", default="", help="provider_id:price1,price2 e.g. p_pl_1:1200,1350")
    a = ap.parse_args()
    if hasattr(__import__('sys').stdout, 'reconfigure'):
        __import__('sys').stdout.reconfigure(encoding='utf-8')
    quote = {}
    if a.price_drift:
        pid, prices = a.price_drift.split(":")
        quote[pid] = [int(x) for x in prices.split(",")]
    clock = Clock(offset=DEMO_START - time.time())   # pretend it is Monday 2 Nov 2026, 08:00
    api = ProviderAPI(Scenario(seed=a.seed, book_script=[x for x in a.book_script.split(",") if x], quote_script=quote), clock)
    agent = Agent(user_id=a.user, session_id=a.session, api=api, model=RuleModel(), store=Store(a.db), clock=clock,
                  trace_path=f"traces/{a.session}.jsonl")
    print(f"session {a.session}  trace: traces/{a.session}.jsonl   (Ctrl-D to quit)")
    rec = agent.recover()
    if rec:
        print("agent>", rec)
    while True:
        try:
            text = input("you> ").strip()
        except EOFError:
            break
        if text:
            print("agent>", agent.handle(text))


if __name__ == "__main__":
    main()
