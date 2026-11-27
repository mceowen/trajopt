"""Rerun the selected configuration of each transcription three times in fresh processes and keep the median solve time."""
import json, os, subprocess, sys
import numpy as np

PY = sys.executable
STEPS = [str(v) for v in (0.25, 0.0125, 0.5, 0.025, 1, 0.05, 2, 0.1, 4, 0.2)]  # the pairs of the screen
sel = json.load(open("plots/matrix/selected.json"))
out = {}
for method, row in sel.items():
    tag = row["tag"]; d = f"plots/final/{tag.replace('|', '_')}"; os.makedirs(d, exist_ok=True)
    times, rows = [], []
    for rep in range(3):
        if os.path.exists(f"{d}/rep{rep}/screen.json"):
            r = json.load(open(f"{d}/rep{rep}/screen.json"))[0]; rows.append(r); times.append(r["scp_time_s"]); continue
        subprocess.run([PY, "matrix.py", "--methods", method, "--only", tag, "--out", f"{d}/rep{rep}", "--samples", "512", "--save-iters", "--steps", *STEPS], check=True,
                       stdout=open(f"{d}/rep{rep}.log", "w"), stderr=subprocess.STDOUT)
        r = json.load(open(f"{d}/rep{rep}/screen.json"))[0]; rows.append(r); times.append(r["scp_time_s"])
    r = rows[int(np.argsort(times)[1])]; r["scp_time_median_s"] = float(np.median(times)); r["scp_times_s"] = times; r["rep_dir"] = f"{d}/rep{int(np.argsort(times)[1])}"
    out[method] = r
    print(f"FINAL {method} {tag} times={np.round(times, 2)} median={r['scp_time_median_s']:.2f} cost={r.get('cost')} R={r.get('R')} limiting={r.get('limiting')} pass={r['pass']}")
json.dump(out, open("plots/final/final.json", "w"), indent=1, default=str)
