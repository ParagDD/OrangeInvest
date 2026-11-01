"""Preference memory with provenance.

* source = 'stated' (user said it) or 'inferred' (agent concluded it).
* Inferred facts are never *used* until the user confirms them.
* forget() deletes the value and leaves a tombstone holding only a hash, so the same inferred value
  can never be re-proposed, and nothing forgotten is ever returned by usable()/view().
"""
import hashlib
import json

KEYS = ("area", "earliest_time", "preferred_technician")


def _h(v):
    return hashlib.sha256(json.dumps(v).encode()).hexdigest()[:16]


class Memory:
    def __init__(self, store, tracer=None):
        self.s, self.tracer = store, tracer

    def _t(self, kind, **kw):
        if self.tracer:
            self.tracer.event(kind, **kw)

    def view(self, user_id):
        rows = self.s.db.execute("SELECT * FROM memory WHERE user_id=? AND forgotten=0 ORDER BY key", (user_id,)).fetchall()
        return [{"key": r["key"], "value": json.loads(r["value"]), "source": r["source"], "confirmed": bool(r["confirmed"])}
                for r in rows]

    def usable(self, user_id):
        return {f["key"]: f["value"] for f in self.view(user_id) if f["source"] == "stated" or f["confirmed"]}

    def pending_inferred(self, user_id):
        return {f["key"]: f["value"] for f in self.view(user_id) if f["source"] == "inferred" and not f["confirmed"]}

    def set_stated(self, user_id, key, value):
        if key not in KEYS:
            raise ValueError(key)
        self.s.db.execute(
            "INSERT INTO memory (user_id,key,value,source,confirmed,forgotten,value_hash) VALUES (?,?,?,?,1,0,?) "
            "ON CONFLICT(user_id,key) DO UPDATE SET value=excluded.value, source='stated', confirmed=1, "
            "forgotten=0, value_hash=excluded.value_hash",
            (user_id, key, json.dumps(value), "stated", _h(value)))
        self._t("memory_write", key=key, source="stated")

    def propose_inferred(self, user_id, key, value):
        """Store an inference as UNCONFIRMED. Refused if forgotten before, or if a stated value exists."""
        row = self.s.db.execute("SELECT * FROM memory WHERE user_id=? AND key=?", (user_id, key)).fetchone()
        if row:
            if row["forgotten"] and row["value_hash"] == _h(value):
                self._t("memory_inference_refused", key=key, reason="tombstoned")
                return False
            if not row["forgotten"]:
                return False  # already have something (stated or inferred): don't overwrite
        self.s.db.execute(
            "INSERT INTO memory (user_id,key,value,source,confirmed,forgotten,value_hash) VALUES (?,?,?,'inferred',0,0,?) "
            "ON CONFLICT(user_id,key) DO UPDATE SET value=excluded.value, source='inferred', confirmed=0, "
            "forgotten=0, value_hash=excluded.value_hash",
            (user_id, key, json.dumps(value), _h(value)))
        self._t("memory_write", key=key, source="inferred", confirmed=False)
        return True

    def confirm(self, user_id, key):
        cur = self.s.db.execute("UPDATE memory SET confirmed=1 WHERE user_id=? AND key=? AND forgotten=0", (user_id, key))
        self._t("memory_confirm", key=key, ok=cur.rowcount == 1)
        return cur.rowcount == 1

    def forget(self, user_id, key):
        row = self.s.db.execute("SELECT * FROM memory WHERE user_id=? AND key=? AND forgotten=0", (user_id, key)).fetchone()
        if not row:
            return False
        self.s.db.execute("UPDATE memory SET value=NULL, forgotten=1, confirmed=0 WHERE user_id=? AND key=?", (user_id, key))
        self._t("memory_forget", key=key)
        return True
