"""Nodal path-limit margin: solve the selected configurations with the heat-rate, dynamic-pressure, and load limits
tightened by a fraction m at the nodes, and replay against the original limits."""
import argparse, json, os, subprocess, sys
import yaml

PY = sys.executable
STEPS = [str(v) for v in (0.25, 0.0125, 0.5, 0.025, 1, 0.05, 2, 0.1, 4, 0.2)]
LIMITS = {"max_heat_rate": 6.0, "max_dynamic_pressure": 15.0, "max_aero_load": 2.5}

ap = argparse.ArgumentParser(); ap.add_argument("--margins", nargs="*", type=float, default=[0.02, 0.04, 0.06])
ap.add_argument("--methods", nargs="*", default=["MS-term", "p-term", "hp-term"]); ap.add_argument("--samples", type=int, default=512)
args = ap.parse_args()
sel = json.load(open("plots/matrix/selected.json"))
os.makedirs("plots/margin", exist_ok=True)
for m in args.margins:
    mission = f"plots/margin/mission_m{m:g}.yaml"; compare = f"plots/margin/compare_m{m:g}.yaml"
    yaml.safe_dump({"inherit": os.path.relpath("mission_all.yaml", "plots/margin"),
                    "constraints": {k: {"upper": [v * (1.0 - m)]} for k, v in LIMITS.items()}}, open(mission, "w"), sort_keys=False)
    base = yaml.safe_load(open("compare_all.yaml"))
    base["trajectory"]["inherit"] = os.path.basename(mission); base["method"]["inherit"] = os.path.relpath("method.yaml", "plots/margin")
    yaml.safe_dump(base, open(compare, "w"), sort_keys=False)
    for method in args.methods:
        tag = sel[method]["tag"]; out = f"plots/margin/m{m:g}/{tag.replace('|', '_')}"
        if os.path.exists(f"{out}/screen.json"): continue
        subprocess.run([PY, "matrix.py", "--methods", method, "--only", tag, "--config", compare, "--limits-config", "compare_all.yaml",
                        "--out", out, "--samples", str(args.samples), "--save-iters", "--steps", *STEPS],
                       stdout=open(f"{out}.log", "w") if os.makedirs(os.path.dirname(out), exist_ok=True) is None else None, stderr=subprocess.STDOUT)
        r = json.load(open(f"{out}/screen.json"))[0]
        print(f"MARGIN {m:g} {tag} conv={r['converged']} it={r['iterations']} cost={r.get('cost')} R={r.get('R')} limiting={r.get('limiting')} pass={r['pass']}", flush=True)
