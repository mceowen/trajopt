"""Select the finish configuration of each handoff and rerun it three times in fresh processes."""
import glob, json, os, subprocess, sys
import numpy as np
PY = sys.executable
rows = [json.load(open(f)) for f in glob.glob("plots/handoff/*/row.json")]
out = {}
for m, seed in (("p-term>MS-term", "p"), ("hp-term>MS-term", "hp")):
    rs = [r for r in rows if r["method"] == m and r.get("R") is not None]
    passing = [r for r in rs if r["pass"]]; conv = [r for r in rs if r["converged"]] or rs
    if passing: best = max(passing, key=lambda r: r["cost"])
    else:
        rmin = min(r["R"] for r in conv); best = min([r for r in conv if r["R"] <= 1.01 * rmin], key=lambda r: r["iterations"])
    d = f"plots/handoff_final/{best['tag'].replace('|', '_').replace('>', '_to_')}"; times = []
    for rep in range(3):
        o = f"{d}/rep{rep}"; rf = glob.glob(f"{o}/*/row.json")
        if not rf:
            subprocess.run([PY, "handoff.py", "--seed", seed, "--finish-nodes", str(best["N"]), "--sz", str(best["sz"]), "--su", str(best["su"]), "--out", o, "--samples", "512", "--save-iters"] + (["--weights"] if best.get("weights") else []),
                           check=True, stdout=open(f"{d}_rep{rep}.log", "w") if os.makedirs(d, exist_ok=True) is None else None, stderr=subprocess.STDOUT)
            rf = glob.glob(f"{o}/*/row.json")
        r = json.load(open(rf[0])); r["rep_dir"] = os.path.dirname(rf[0]); times.append(r["scp_time_s"]); last = r
    last["scp_time_median_s"] = float(np.median(times)); last["scp_times_s"] = times
    last.update(screened=len(rs), converged_count=len([r for r in rs if r["converged"]]), passing=len(passing)); out[m] = last
    print(f"FINAL {m} {best['tag']} times={np.round(times, 2)} median={last['scp_time_median_s']:.2f} cost={last.get('cost')} R={last.get('R')} limiting={last.get('limiting')} pass={last['pass']} ({len(passing)} passing of {len(rs)})")
os.makedirs("plots/handoff_final", exist_ok=True); json.dump(out, open("plots/handoff_final/final.json", "w"), indent=1, default=str)
