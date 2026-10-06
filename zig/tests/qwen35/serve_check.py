"""Exactness of a running native server over HTTP: drafted == serial, solo == together, resumed == fresh.

usage: serve_check.py URL MODEL [TOKENS]
"""

import json
import sys
import threading
import urllib.request

URL, MODEL = sys.argv[1], sys.argv[2]
TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 64
PROMPTS = [
    "Write a short poem about the sea.",
    "List the first twelve prime numbers, separated by commas.",
    "Explain in two sentences why the sky is blue.",
    "Count from one to thirty in words.",
]


def ask(messages, draft=True, temperature=0.0, seed=7):
    body = {"model": MODEL, "messages": messages, "max_tokens": TOKENS, "temperature": temperature, "seed": seed,
            "draft": draft, "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as reply:
        doc = json.load(reply)
    return doc["choices"][0]["message"]["content"], doc.get("usage", {}).get("prompt_tokens_details", {})


def user(text):
    return [{"role": "user", "content": text}]


def together(draft, temperature):
    out = [None] * len(PROMPTS)

    def run(i):
        out[i] = ask(user(PROMPTS[i]), draft, temperature)[0]

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(PROMPTS))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


failed = 0


def check(name, same):
    global failed
    failed += not same
    print(("ok   " if same else "FAIL ") + name, flush=True)


for label, temperature in (("greedy", 0.0), ("t0.8", 0.8)):
    serial = [ask(user(p), False, temperature)[0] for p in PROMPTS]
    drafted = [ask(user(p), True, temperature)[0] for p in PROMPTS]
    check(f"{label}: drafted == serial", drafted == serial)
    check(f"{label}: solo == together", together(True, temperature) == drafted)
    check(f"{label}: serial together == serial solo", together(False, temperature) == serial)
    again = [ask(user(p), True, temperature)[0] for p in PROMPTS]
    check(f"{label}: the same twice", again == drafted)
first, _ = ask(user(PROMPTS[0]), True)
turn = user(PROMPTS[0]) + [{"role": "assistant", "content": first}, {"role": "user", "content": "Now say it shorter."}]
fresh, _ = ask(turn, False)
resumed, details = ask(turn, True)
check(f"resumed == fresh (cached {details})", resumed == fresh)
print("failed", failed)
sys.exit(1 if failed else 0)
