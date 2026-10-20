#!/usr/bin/env python3
"""Validate actual benchmark schemas and ordered A/B receipts; no GPU work."""
import json, math, sys, re, statistics

def require(ok, why):
    if not ok: raise ValueError(why)

def receipt(r, tokens):
    require(r.get("complete") is True and not r.get("error"), "incomplete/error receipt")
    require(r.get("tokens") == tokens, "wrong completion token count")
    require(bool(r.get("token_sha")), "missing token_sha")
    require(bool(re.fullmatch(r"[0-9a-f]{64}", r.get("output_sha256", ""))), "missing complete reply output_sha256")
    return (r["token_sha"], r["output_sha256"])

def read(path, mode, n):
    with open(path) as f: d=json.load(f)
    if mode == "decode":
        require(isinstance(d,list) and len(d)==2, "expected two decode prompt rows")
        hashes={}
        for r in d:
            key=(r["prompt"],r["temperature"])
            require(key not in hashes and key[1]==0, "duplicate prompt or wrong temperature")
            runs=r.get("evidence",[])
            require(len(runs)==3, "expected three evidence receipts")
            hashes[key]=[receipt(x,256) for x in runs]
            require(len(set(hashes[key]))==1, "greedy decode outputs differ within server")
            rates=r.get("decode_tps_all",[])
            require(len(rates)==3 and all(isinstance(x,(float,int)) and math.isfinite(x) and x>0 for x in rates), "missing/invalid decode rates")
            require(isinstance(r.get("decode_tps_median"),(int,float)) and math.isfinite(r["decode_tps_median"]) and r["decode_tps_median"]>0, "invalid median")
            # Original client stores per-repetition rates rounded to two decimals.
            require(round(r["decode_tps_median"], 2) == statistics.median(rates), "reported median disagrees with rounded repetition median")
            print(r["prompt"], "decode_tps", r["decode_tps_median"], "all", rates)
        require(set(hashes)=={("fibonacci-raw",0),("gpu-chat-no-think",0)}, "unexpected prompt set")
        return hashes
    require(d.get("qualified") is True, "cold benchmark not qualified")
    rows=d.get("rows",[])
    require(len(rows)==3 and [r.get("phase") for r in rows]==["first","repeated","concurrent"], "unexpected cold phases")
    hashes=[receipt(r,16) for r in rows]
    for r in rows:
        require(r.get("prompt_tokens")==n and r.get("cached_tokens")==0 and r.get("cache_state")=="cold", "missing cold prompt evidence")
        require(isinstance(r.get("ttft_s"),(int,float)) and math.isfinite(r["ttft_s"]) and r["ttft_s"]>0, "invalid ttft")
    require(len(set(hashes))==1, "cold outputs differ within server")
    r=rows[0]
    print("cold",n,"ttft",r["ttft_s"],"prefill_tps",n/r["ttft_s"])
    return {"fixture":d["fixture_sha256"],"hashes":hashes}

if __name__=="__main__":
    try:
        mode,n,*paths=sys.argv[1:]
        require(len(paths) in (1,4), "expected one receipt or old,new,new,old quartet")
        results=[read(p,mode,int(n)) for p in paths]
        require(all(x==results[0] for x in results), "TOKEN_SHA_MISMATCH across A/B quartet")
        print("VALID" if len(paths)==1 else "PASS token_sha equal across old,new,new,old")
    except Exception as e:
        sys.exit("FAIL: "+str(e))
