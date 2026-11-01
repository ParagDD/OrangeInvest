"""Local web console (stdlib only):  python -m errand_agent.webapp  ->  http://127.0.0.1:8765
Chat on the left, live trace / approval ledger / provider ground truth on the right."""
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .agent import Agent
from .clock import FakeClock
from .errors import SimulatedCrash
from .mocks import ProviderAPI, Scenario
from .model import RuleModel
from .store import Store

ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static" / "index.html"
RUNS = ROOT_DIR / "traces" / "web"


class Console:
    def __init__(self):
        self.lock = threading.RLock()
        self.new({})

    def new(self, opt):
        with self.lock:
            RUNS.mkdir(parents=True, exist_ok=True)
            self.sid = f"web-{int(time.time())}"
            self.user, self.crashed = "u1", False
            self.clock = FakeClock()
            quote = {"p_pl_1": [1200, 1350]} if opt.get("drift") else {}
            self.api = ProviderAPI(Scenario(book_script=[x for x in opt.get("script", "").split(",") if x], quote_script=quote,
                                            evasive_injection=bool(opt.get("evasive")),
                                            cancel_always_503=bool(opt.get("cancel503"))), self.clock)
            self.db = RUNS / f"{self.sid}.db"
            self.trace = RUNS / f"{self.sid}.jsonl"
            self.store = None
            self._mk()

    def _mk(self):
        if self.store:
            self.store.close()
        self.store = Store(self.db)
        self.agent = Agent(user_id=self.user, session_id=self.sid, api=self.api, model=RuleModel(), store=self.store,
                           clock=self.clock, trace_path=self.trace)

    def chat(self, text):
        with self.lock:
            if self.crashed:
                return {"reply": "The agent process is down. Press Restart to bring it back.", "crashed": True}
            try:
                return {"reply": self.agent.handle(text), "crashed": False}
            except SimulatedCrash:
                self.crashed = True
                return {"reply": None, "crashed": True}

    def restart(self):
        with self.lock:
            self._mk()
            self.crashed = False
            return {"reply": self.agent.recover(), "crashed": False}

    def arm_crash(self, point):
        self.agent.executor.hooks.crash_at = point or None

    def trace_since(self, i):
        if not self.trace.exists():
            return []
        lines = self.trace.read_text(encoding="utf-8").splitlines()[i:]
        return [{"i": i + k, **json.loads(l)} for k, l in enumerate(lines)]

    def snapshot(self):
        with self.lock:
            s, db = self.agent.s, self.store.db
            rows = lambda q: [dict(r) for r in db.execute(q).fetchall()]
            return {"sid": self.sid, "crashed": self.crashed, "state": s.state, "action": s.action,
                    "slots": s.slots, "options": s.options, "pending": s.pending, "meta": s.pending_meta,
                    "shown": s.shown_digest, "hold": s.hold, "clock": self.clock.now(), "result": s.result,
                    "memory": self.agent.memory.view(self.user), "turns": s.turns,
                    "budget": {"model": self.agent.budget.model_calls, "tool": self.agent.budget.tool_calls},
                    "approvals": rows("SELECT approval_id,digest,status,seq,reason,booking_id FROM approvals ORDER BY seq"),
                    "bookings": list(self.api.bookings.values()),
                    "creations": self.api.creations, "provider_calls": len(self.api.calls),
                    "provider_mutations": sum(c["tool"] in ("book", "cancel", "reschedule") for c in self.api.calls),
                    "approved_mutations": db.execute("SELECT COUNT(*) FROM mutations").fetchone()[0]}


C = Console()


class H(BaseHTTPRequestHandler):
    def _send(self, obj, code=200, ctype="application/json"):
        body = obj if isinstance(obj, bytes) else json.dumps(obj, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path, _, q = self.path.partition("?")
        if path == "/":
            if not STATIC.exists():
                return self._send(b"Console UI not found", 404, "text/plain")
            return self._send(STATIC.read_bytes(), ctype="text/html")
        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if path == "/api/state":
            return self._send(C.snapshot())
        if path == "/api/trace":
            since = 0
            if "=" in q:
                val = q.split("=")[1]
                if val.isdigit():
                    since = int(val)
            return self._send(C.trace_since(since))
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            d = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            d = {}
        p = self.path
        if p == "/api/chat":
            return self._send(C.chat(str(d.get("text", ""))[:500]))
        if p == "/api/new":
            C.new(d)
            return self._send(C.snapshot())
        if p == "/api/advance":
            try:
                sec = float(d.get("seconds", 0))
            except (ValueError, TypeError):
                sec = 0.0
            C.clock.advance(sec)
            return self._send(C.snapshot())
        if p == "/api/restart":
            return self._send(C.restart())
        if p == "/api/crash":
            C.arm_crash(d.get("point"))
            return self._send({"armed": d.get("point")})
        self._send({"error": "not found"}, 404)

    def log_message(self, *a):
        pass


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"Errand Agent console: http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


if __name__ == "__main__":
    main()
