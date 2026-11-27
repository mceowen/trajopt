"""Hybrid methods on the bare powered-descent problem: a p- or hp-method seed stage, converged or stopped at an iteration budget,
handed off as the initial guess of the 15-node multiple-shooting final stage."""
import argparse, json, os
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np, yaml
from trajopt.trajectory_analyzer import TrajectoryAnalyzer
from trajopt.analysis.experiments import ExperimentCase, ExperimentVariant, run_experiment
from trajopt.methods.stable.algflows import AlgorithmFlow
from pdg_matrix import row_of

ap = argparse.ArgumentParser(); ap.add_argument("--seed", choices=["p", "hp", "hp-term"], required=True)
ap.add_argument("--out", default="plots/pdg_handoff"); ap.add_argument("--mission", default="mission_nostc.yaml")
ap.add_argument("--seed-budget", type=int, default=None); ap.add_argument("--guess-json", default=None); ap.add_argument("--case", default=None)
args = ap.parse_args()
tag = f"{args.seed}>MS" + (f"|b{args.seed_budget}" if args.seed_budget else "") + (f"|{args.case}" if args.case else "")
work = os.path.join(args.out, tag.replace("|", "_").replace(">", "_to_")); os.makedirs(work, exist_ok=True)
seed = yaml.safe_load(open(f"stages/{'p_method' if args.seed == 'p' else 'hp_method'}.yaml")); seed.pop("inherit", None)
if args.seed == "hp-term": seed.setdefault("method", {}).setdefault("hyperparams", {})["flags"] = {"zoh_dilation": 1}
seed.setdefault("trajectory", {})["inherit"] = os.path.relpath(args.mission, work)
m = seed.setdefault("method", {}); inh = m.get("inherit", [])
inh = ([os.path.relpath("method.yaml", work), "trajopt/methods/stable/configs/first_order/autoscvx-term-ps.yaml"] if args.seed in ("p", "hp-term") else [os.path.relpath("method.yaml", work), "trajopt/methods/common/configs/discretize/pseudospectral_p.yaml"])
m["inherit"] = inh; m["history"] = "full"
if args.seed_budget: m.setdefault("hyperparams", {})["convergence"] = {"iter_max": int(args.seed_budget)}
if args.guess_json: m.setdefault("hyperparams", {})["initial_guess"] = {"default": dict(t_start=0.0, t_stop=1.0, x_stop=[1.0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0], **json.loads(args.guess_json))}
finish = {"trajectory": {"inherit": os.path.relpath(args.mission, work)}, "method": {"inherit": os.path.relpath("method.yaml", work), "history": "full", "hyperparams": {"flags": {"quiet": True}}}}
yaml.safe_dump(seed, open(os.path.join(work, "seed.yaml"), "w"), sort_keys=False); yaml.safe_dump(finish, open(os.path.join(work, "finish.yaml"), "w"), sort_keys=False)
flow_path = os.path.join(work, "flow.yaml")
yaml.safe_dump({"flow": {"stages": [{"name": "seed", "config": "seed.yaml", **({"allow_nonconverged": True} if args.seed_budget else {})}, {"name": "finish", "config": "finish.yaml", "initial_guess": {"from": "seed"}}]}}, open(flow_path, "w"), sort_keys=False)
traj = TrajectoryAnalyzer("pdg_compare.yaml"); cfg = traj.config
variant = ExperimentVariant(tag, {}, operation_factory=lambda config: AlgorithmFlow(flow_path), operation_id=f"Flow:{tag}")
exp = run_experiment([ExperimentCase("nominal", cfg, nominal=True)], [variant], evaluate=True, analysis_config=cfg.analysis)
run = exp.runs[0]; flow = run.result
stages = {} if flow is None or not hasattr(flow, "stages") else {n: dict(converged=bool(st.converged), iterations=int(getattr(st.output, "iterations", 0) or 0) if st.output is not None else 0,
            solve_s=float(dict(getattr(st.output, "timings", {}) or {}).get("total_s", float("nan"))) if st.output is not None else float("nan"), status=str(st.status)) for n, st in flow.stages.items()}
row = row_of(run, dict(method=f"{args.seed}>MS", tag=tag, stages=stages, seed_budget=args.seed_budget))
row["iterations"] = sum(s["iterations"] for s in stages.values()); row["scp_time_s"] = float(np.nansum([s["solve_s"] for s in stages.values()]))
row["converged"] = bool(stages.get("finish", {}).get("converged", False)); row["pass"] = bool(row["converged"] and row.get("R") is not None and row["R"] <= 1.0)
json.dump(row, open(os.path.join(work, "row.json"), "w"), indent=1, default=str)
print(f"RESULT {tag} stages={ {n: (s['converged'], s['iterations'], round(s['solve_s'], 1)) for n, s in stages.items()} } tF={row.get('cost')} R={row.get('R')} limiting={row.get('limiting')} pass={row['pass']}", flush=True)
