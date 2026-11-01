import time
from datetime import datetime, timezone

DEMO_START = datetime(2026, 11, 2, 8, 0, tzinfo=timezone.utc).timestamp()  # Monday 08:00


class Clock:
    """Wall clock with an optional offset (used by the CLI so the demo dataset's dates line up)."""
    def __init__(self, offset=0.0):
        self.offset = offset

    def now(self):
        return time.time() + self.offset

    def sleep(self, s):
        time.sleep(s)

    def today_iso(self):
        return datetime.fromtimestamp(self.now(), tz=timezone.utc).date().isoformat()

    def hhmm(self, ts):
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M")


class FakeClock(Clock):
    """Deterministic clock: only moves when told to (or when a tool 'takes time')."""
    def __init__(self, start=DEMO_START):
        super().__init__()
        self.t = start

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def advance(self, s):
        self.t += s
