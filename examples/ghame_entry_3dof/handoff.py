"""Two-stage handoffs: a converged collocation seed stage resampled as the initial guess of a multiple-shooting final stage, optionally passing its terminal penalty weights."""
import argparse, copy, json, os
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import yaml
from trajopt.trajectory_analyzer import TrajectoryAnalyzer
from trajopt.analysis.experiments import ExperimentCase, ExperimentVariant, run_experiment
from trajopt.methods.stable.algflows import AlgorithmFlow
from matrix import limits_from, SCALES
from replay import replay, PATH_ALLOW

ap = argparse.ArgumentParser(); ap.add_argument("--seed", choices=["p", "hp"], required=True)
ap.add_argument("--finish-nodes", type=int, default=49); ap.add_argument("--sz", type=float, default=1.0); ap.add_argument("--su", type=float, default=0.05)
ap.add_argument("--out", default="plots/handoff"); ap.add_argument("--samples", type=int, default=64); ap.add_argument("--save-iters", action="store_true"); ap.add_argument("--weights", action="store_true"); ap.add_argument("--mission", default="mission_all.yaml")
ap.add_argument("--seed-nodes", type=int, default=None); ap.add_argument("--seed-segments", type=int, default=None)
ap.add_argument("--seed-budget", type=int, default=None, help="stop the seed stage after this many iterations and hand off even if not converged")
ap.add_argument("--guess", type=float, nargs=5, default=None, metavar=("BANK0", "AOA0", "BANK1", "AOA1", "TSTOP"), help="initial guess of the seed stage: u_start, u_stop, t_stop")
ap.add_argument("--case", default=None, help="label of the dispersed case for the tag")
ap.add_argument("--finish-method", default="method.yaml", help="method config of the final stage (method_std.yaml for the standard setting)")
ap.add_argument("--seed-method", default="method.yaml", help="method config of the seed stage (method_term_sl.yaml for a straight-line guess)")
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
tag = f"{args.seed}-term>MS-term|N{args.finish_nodes}|sz{args.sz:g}" + (f"|s{args.seed_nodes}" if args.seed_nodes else "") + (f"|b{args.seed_budget}" if args.seed_budget else "") + ("|w" if args.weights else "") + (f"|{args.case}" if args.case else "") + ("|std" if args.finish_method != "method.yaml" else "") + ("|sl" if args.seed_method != "method.yaml" else "") + ("" if args.mission == "mission_all.yaml" else "|" + os.path.splitext(os.path.basename(args.mission))[0])
work = os.path.join(args.out, tag.replace("|", "_").replace(">", "_to_")); os.makedirs(work, exist_ok=True)
finish = yaml.safe_load(open("stages/ms_finish.yaml")); finish["trajectory"]["num_nodes"] = args.finish_nodes
finish["trajectory"]["inherit"] = os.path.relpath(args.mission, work)
finish["method"]["inherit"] = os.path.relpath(args.finish_method, work)
finish["method"]["hyperparams"]["trust_region"]["tr_step"] = {"z": {"default": args.sz}, "nu": {"default": args.su, "u": args.su, "s": 1.0}}
finish["method"]["history"] = "full"
seed = yaml.safe_load(open(f"stages/{args.seed}_seed.yaml")); seed["trajectory"]["inherit"] = os.path.relpath(args.mission, work)
if args.seed_nodes: seed["trajectory"]["num_nodes"] = args.seed_nodes
if args.seed_segments: seed["trajectory"]["hp_segments"] = args.seed_segments
if args.seed_budget: seed["method"]["hyperparams"]["convergence"] = {"iter_max": int(args.seed_budget)}
if args.guess: seed["method"]["hyperparams"]["initial_guess"] = {"default": {"t_start": 0.0, "t_stop": float(args.guess[4]), "u_start": [float(args.guess[0]), float(args.guess[1])], "u_stop": [float(args.guess[2]), float(args.guess[3])]}}
seed["method"]["inherit"][0] = os.path.relpath(args.seed_method, work)
seed["method"]["history"] = "full"
yaml.safe_dump(seed, open(os.path.join(work, "seed.yaml"), "w"), sort_keys=False); yaml.safe_dump(finish, open(os.path.join(work, "finish.yaml"), "w"), sort_keys=False)
flow_path = os.path.join(work, "flow.yaml")
yaml.safe_dump({"flow": {"stages": [{"name": "seed", "config": "seed.yaml", **({"allow_nonconverged": True} if args.seed_budget else {})}, {"name": "finish", "config": "finish.yaml", "initial_guess": {"from": "seed", "endpoint_weights": bool(args.weights)}}]}}, open(flow_path, "w"), sort_keys=False)

traj = TrajectoryAnalyzer("compare_all.yaml"); cfg = traj.config
limits = limits_from(cfg, list(traj.problem.phases.values())[0].build())
variant = ExperimentVariant(tag, {}, operation_factory=lambda config: AlgorithmFlow(flow_path), operation_id=f"Flow:{tag}")
exp = run_experiment([ExperimentCase("nominal", cfg, nominal=True)], [variant], evaluate=True, analysis_config=cfg.analysis)
run = exp.runs[0]; flow = run.result; r = run.solve_result
stages = {} if flow is None or not hasattr(flow, "stages") else {n: dict(converged=bool(st.converged), iterations=int(getattr(st.output, "iterations", 0) or 0) if st.output is not None else 0,
            solve_s=float(dict(getattr(st.output, "timings", {}) or {}).get("total_s", float("nan"))) if st.output is not None else float("nan"), status=str(st.status)) for n, st in flow.stages.items()}
row = dict(method=f"{args.seed}-term>MS-term", tag=tag, weights=bool(args.weights), N=args.finish_nodes, K=1, control="foh", stride=1, sz=args.sz, su=args.su,
           seed_N=seed["trajectory"]["num_nodes"], seed_K=seed["trajectory"].get("hp_segments", 1), stages=stages,
           converged=bool(getattr(r, "converged", False)), iterations=sum(s["iterations"] for s in stages.values()),
           scp_time_s=sum(s["solve_s"] for s in stages.values()), wall_s=float(run.wall_s), error=run.error)
if run.evaluation is not None and run.evaluation.phases:
    rep = replay(run, limits, SCALES, "ms", 1, "foh", 1, samples=args.samples)
    row.update(R=rep["R"], limiting=rep["limiting"], cost=rep["cost"], nodal_cost=rep["nodal_cost"], t_final=rep["t_final"], checks=rep["checks"], path_allowance=PATH_ALLOW)
    np.savez(os.path.join(work, "replay.npz"), **{f"dense_{k}": v for k, v in rep["dense"].items() if k != "path"}, **{f"path_{k.replace(' ', '_')}": v for k, v in rep["dense"]["path"].items()}, **{f"nodes_{k}": v for k, v in rep["nodes"].items()})
    if args.save_iters and getattr(r, "phase_data", None):
        its = [d for d in next(iter(r.phase_data.values())).iter_data_list if d.get("chk") is not None]
        hist = {k: np.array([float(d["chk"].get(k, 0.0)) for d in its]) for k in ("dz", "final_state", "nonconvex_inequality", "dynamics")}
        hist["objective"] = np.array([float(d.get("x_opt")[-1][2]) if d.get("x_opt") is not None else np.nan for d in its])
        np.savez(os.path.join(work, "iters.npz"), **hist)
row["pass"] = bool(row["converged"] and row.get("R") is not None and row["R"] <= 1.0)
json.dump(row, open(os.path.join(work, "row.json"), "w"), indent=1, default=str)
print(f"RESULT {tag} stages={ {n: (s['converged'], s['iterations'], round(s['solve_s'], 1)) for n, s in stages.items()} } cost={row.get('cost')} R={row.get('R')} limiting={row.get('limiting')} pass={row['pass']}")
