"""One JSONL trace per run (session). Everything passes through redact(): no secrets, bounded strings."""
import json
import os

from .sanitize import redact


class Tracer:
    def __init__(self, path, clock, run_id):
        self.path, self.clock, self.run_id = str(path), clock, run_id
        self.n = 0
        self.turn = 0
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

    def event(self, kind, **data):
        self.n += 1
        rec = {"n": self.n, "t": round(self.clock.now(), 3), "run": self.run_id, "turn": self.turn, "kind": kind}
        rec.update(redact(data))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
