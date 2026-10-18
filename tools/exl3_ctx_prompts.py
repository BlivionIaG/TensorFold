#!/usr/bin/env python3
"""Prompt sets for `tf-cuda-test exl3-ctx` (ASCII text tokenised with exl3_tok, plus qbench ids from the pack).

  exl3_ctx_prompts.py PACK_DIR OUT_DIR

Writes OUT_DIR/tf.txt (one comma separated id list per line, six varied prompts), OUT_DIR/long.txt (a ~2250 token needle
prompt, the needle early, the question last), OUT_DIR/needle_answer.txt (expected ids of the answer) and OUT_DIR/prefix_{a,b,c}.txt.
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("exl3_tok", os.path.join(os.path.dirname(os.path.abspath(__file__)), "exl3_tok.py"))
src = open(spec.origin).read().replace("\nmain()\n", "\n")
tok = {}
exec(compile(src, "exl3_tok", "exec"), tok)

pack, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
vocab, ranks = tok["load"](pack)
b2u = tok["bytes_to_unicode"]()
PAT = tok["PAT"]

def encode(text):
    ids = []
    for piece in PAT.findall(text):
        word = "".join(b2u[b] for b in piece.encode())
        ids += [vocab[p] for p in tok["bpe"](word, ranks)]
    return ids

assert encode(" Paris") == [11751], encode(" Paris")
assert encode("The capital of France is Paris.") == [760, 6511, 314, 9338, 369, 11751, 13]

texts = [
    ("numbers", "Here is a list of numbers: 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144. Each number is the sum of the two before it, so the next one is"),
    ("prose_short", "The old lighthouse keeper climbed the spiral stairs every evening, lit the great lamp, and watched the ships pass in the dark water below."),
    ("code", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n\nprint(fibonacci(10))\n# The output of this program is"),
    ("prose_long", "The water cycle describes how water moves between the oceans, the atmosphere and the land. Heat from the sun evaporates water from the surface of the sea, and the vapour rises, cools and condenses into clouds. When the droplets grow heavy enough they fall as rain or snow. Some of that water runs into rivers and returns to the sea, while some soaks into the soil and feeds the groundwater that plants and wells depend on. Scientists who study this cycle measure rainfall, river flow and soil moisture to predict droughts and floods. In a warming climate the cycle speeds up, which means that"),
    ("qa", "Question: What is the boiling point of water at sea level in degrees Celsius?\nAnswer: Water boils at 100 degrees Celsius at sea level.\nQuestion: Who wrote the play Hamlet?\nAnswer:"),
]
qb = json.load(open(os.path.join(pack, "qbench_prompts.json")))
rows = [v for v in qb.values() if isinstance(v, list)][0]
qids = rows[0]["input_ids"]
lines = []
for name, t in texts:
    ids = encode(t)
    lines.append((name, ids))
lines.append(("qbench_row0", qids))
with open(os.path.join(out, "tf.txt"), "w") as f:
    for name, ids in lines:
        f.write("%s %s\n" % (name, ",".join(map(str, ids))))
        print("tf", name, len(ids))

# long needle prompt
subjects = ["harbor office", "grain store", "river ferry", "weather station", "market hall", "bell tower", "copper mine", "salt works", "post office", "mill"]
goods = ["crates of copper", "barrels of salt", "sacks of barley", "bolts of cloth", "coils of rope", "jars of honey", "sheets of tin", "casks of oil"]
needle = "The secret access code for the vault is 7391. Remember it."
question = "\n\nQuestion: What is the secret access code for the vault?\nAnswer: The secret access code for the vault is"
text = "Daily log of a coastal town.\n" + needle + "\n"
n = 0
while len(encode(text + question)) < 2300:
    n += 1
    s = subjects[(n * 7) % len(subjects)]
    g = goods[(n * 5 + n // 3) % len(goods)]
    text += "On day %d the %s recorded %d %s and %d visitors; the wind came from the %s.\n" % (n, s, 20 + (n * 37) % 90, g, 3 + (n * 11) % 40, ["north", "south", "east", "west"][n % 4])
long_ids = encode(text + question)
open(os.path.join(out, "long.txt"), "w").write(",".join(map(str, long_ids)) + "\n")
ans = encode(" 7391")
open(os.path.join(out, "needle_answer.txt"), "w").write(",".join(map(str, ans)) + "\n")
print("long", len(long_ids), "needle answer ids", ans)

# prefix cache prompts: A shared, B / C continuations
a = "A lighthouse stands on the northern cape, and every evening the keeper climbs its spiral stairs to light the great lamp. Ships far out at sea steer by that light, and the fishing boats come home to the harbor below it before the dark."
b = " The keeper writes the weather in a thick book."
c = " When the fog rolls in, the horn is sounded every thirty seconds, and the"
for nm, t in (("a", a), ("b", b), ("c", c)):
    ids = encode(t)
    open(os.path.join(out, "prefix_%s.txt" % nm), "w").write(",".join(map(str, ids)) + "\n")
    print("prefix", nm, len(ids))
# long A for the sparse-regime prefix test: first 2100 tokens of the needle prompt
open(os.path.join(out, "prefix_long_a.txt"), "w").write(",".join(map(str, long_ids[:2100])) + "\n")
