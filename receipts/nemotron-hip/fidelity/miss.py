import numpy as np, os, sys
A=os.path.expanduser("$REFS"); N=os.path.expanduser("$REFS"); W=os.path.expanduser("$WORK/logs"); V=131072
P={"chat_full":27,"code_full":15,"story_full":15,"chat_long":27,"code_long":15}
for label in sys.argv[1:]:
  for kind in ("dec","pro"):
    ms=[]
    for seq in ("story_full","chat_long","code_long"):
      p=P[seq]; ref=np.load(f"{A}/ref_{seq}.npy" if seq.endswith("long") else f"{N}/ref_{seq}.npy", mmap_mode="r")
      h=np.fromfile(f"{W}/{kind}_{label}_final_{seq}.bin",dtype=np.float32).reshape(-1,V)
      if kind=="pro": h=h[p:]
      r=np.asarray(ref[p:p+len(h)],dtype=np.float64)
      for i in np.where(r.argmax(1)!=h.argmax(1))[0]:
        s=np.sort(r[i]); ms.append(round(float(s[-1]-s[-2]),3))
    print(label, kind, "misses (story_full, chat_long, code_long) ref margins:", ms)
