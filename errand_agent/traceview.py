"""Pretty-print (and optionally follow) a trace file. Use beside the chat for the demo."""
import json
import sys
import time

COLORS = {"user_msg": "\033[36m", "agent_msg": "\033[32m", "state": "\033[33m", "tool_call": "\033[35m",
          "tool_result": "\033[35m", "approval_granted": "\033[1;32m", "approval_voided": "\033[1;31m", "limit": "\033[1;31m",
          "untrusted_flagged": "\033[1;31m", "book_timeout": "\033[31m", "error": "\033[31m"}


def line(e):
    k = e["kind"]
    body = {x: v for x, v in e.items() if x not in ("n", "t", "run", "turn", "kind")}
    if k == "state":
        s = f"{body['frm']} -> {body['to']}  ({body.get('why', '')})"
    elif k in ("user_msg", "agent_msg"):
        s = body["text"].replace("\n", " ⏎ ")[:160]
    elif k == "tool_call":
        s = f"{body['tool']} {json.dumps(body.get('args'))[:120]}"
    elif k == "tool_result":
        s = f"{body['tool']} -> {body.get('outcome')} {body.get('code', '')} {json.dumps(body.get('result', ''))[:100]}"
    else:
        s = json.dumps(body)[:160]
    return f"{COLORS.get(k, '')}[{e['turn']}] {k:18} {s}\033[0m"


def main():
    path, follow = sys.argv[1], "--follow" in sys.argv
    pos = 0
    while True:
        try:
            with open(path) as f:
                f.seek(pos)
                for l in f:
                    print(line(json.loads(l)), flush=True)
                pos = f.tell()
        except FileNotFoundError:
            pass
        if not follow:
            break
        time.sleep(0.4)


if __name__ == "__main__":
    main()
