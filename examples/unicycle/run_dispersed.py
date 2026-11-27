"""Dispersed initial guesses for the algorithm variants, optionally sweeping the dual step."""
import json, os, sys, argparse
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from trajopt.trajectory_analyzer import TrajectoryAnalyzer
from trajopt.common.tools import deep_merge, recursive_attrdict
from trajopt.analysis.experiments import ExperimentVariant, monte_carlo_cases, run_experiment

ap = argparse.ArgumentParser(); ap.add_argument("config", nargs="?", default="dispersed.yaml")
ap.add_argument("--gamma", nargs="*", type=float, default=None); ap.add_argument("--num", type=int, default=20)
ap.add_argument("--methods", nargs="*", default=["Variant 1"], help="variants to sweep over --gamma")
ap.add_argument("--pos", type=float, default=1.0, help="uniform position offset of the guess endpoints [m]")
ap.add_argument("--vel", type=float, default=0.5, help="uniform velocity offset of the guess endpoints")
ap.add_argument("--wp", type=float, default=3.0, help="uniform position offset of the straight-line waypoint [m]")
args = ap.parse_args()
OUT = "plots/" + os.path.splitext(os.path.basename(args.config))[0] + ("" if args.gamma is None else "_gamma")
os.makedirs(OUT, exist_ok=True)
traj = TrajectoryAnalyzer(args.config)
cfg = traj.config
guess = cfg.method.hyperparams.initial_guess.default
if "waypoint" in guess:
    nx = len(guess.waypoint); wp = args.wp
    if nx == 6:
        lb = [-wp, -wp, -0.5 * wp] + [0.0] * 3; ub = [wp, wp, 0.5 * wp] + [0.0] * 3
    else:
        lb = [-wp, -wp, 0.0, 0.0, 0.0]; ub = [wp, wp, 0.0, 0.0, 0.0]
    samples = {"method.hyperparams.initial_guess.default.waypoint": {"type": "uniform", "lb": lb, "ub": ub}}
elif "x_stop" in guess:
    nx = len(guess.x_start)
    if nx == 6:
        lb = [-args.pos] * 3 + [-args.vel] * 3; ub = [args.pos] * 3 + [args.vel] * 3
    else:
        lb = [-args.pos, -args.pos, 0.0, -0.5, 0.0]; ub = [args.pos, args.pos, args.vel, 0.5, 0.0]
    samples = {"method.hyperparams.initial_guess.default.x_start": {"type": "uniform", "lb": lb, "ub": ub},
               "method.hyperparams.initial_guess.default.x_stop": {"type": "uniform", "lb": lb, "ub": ub}}
else:
    nx = len(guess.x_start)
    if nx == 6:
        lb = [-args.pos] * 3 + [-args.vel] * 3; ub = [args.pos] * 3 + [args.vel] * 3
    else:
        lb = [-args.pos, -args.pos, 0.0, -0.5, 0.0]; ub = [args.pos, args.pos, args.vel, 0.5, 0.0]
    samples = {"method.hyperparams.initial_guess.default.x_start": {"type": "uniform", "lb": lb, "ub": ub}}
cfg["variations"] = {"seed": 0, "num": args.num, "samples": samples}
cases = monte_carlo_cases(cfg)
variants = []
for name, overrides in cfg.analysis.methods.items():
    if args.gamma is None:
        variants.append(ExperimentVariant(name, overrides or {}))
    elif name in args.methods:
        for g in args.gamma:
            ov = deep_merge(recursive_attrdict(dict(overrides or {})), recursive_attrdict({"hyperparams": {"penalties": {"default": {"dual": {"tr_step": g}}}}}))
            variants.append(ExperimentVariant(f"{name} (gamma={g:g})", ov))
print(f"{len(cases)} cases x {len(variants)} variants")
exp = run_experiment(cases, variants, evaluate=True, analysis_config=cfg.analysis)
rows = []
for run in exp.runs:
    r, ev = run.solve_result, run.evaluation
    tm = dict(getattr(r, "timings", {}) or {}); it = int(getattr(r, "iterations", 0) or 0)
    mv = cost = tF = objective = None
    if ev is not None and ev.phases:
        ph = next(iter(ev.phases.values())); fin, dg = ph.final, ph.diagnostics
        viol = dg["physical_constraint_violations"]; mv = max((float(v["maximum"]) for v in viol.values()), default=0.0)
        tF = float(np.asarray(fin.t_opt)[-1]); last = next(iter(r.phase_data.values())).iter_data_list[-1]; cost = float(last.get("cost", np.nan)); objective = cost - float(last.get("penalty_cost", 0.0) or 0.0)
    rows.append(dict(case=run.case, variant=run.variant, converged=bool(getattr(r, "converged", False)), status=str(getattr(r, "status", None)),
                     iterations=it, scp_time_s=float(tm.get("total_s", np.nan)), ms_per_iter=(1000 * float(tm.get("total_s", np.nan)) / it if it else np.nan),
                     max_violation=mv, cost=cost, objective=objective, final_time=tF, wall_s=run.wall_s, error=run.error))
json.dump(rows, open(f"{OUT}/runs.json", "w"), indent=1)
def _stat(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))
table = {}
for v in variants:
    sel = [x for x in rows if x["variant"] == v.name]; conv = [x for x in sel if x["converged"]]
    table[v.name] = dict(runs=len(sel), converged=len(conv), iterations=_stat([x["iterations"] for x in conv]), scp_time_s=_stat([x["scp_time_s"] for x in conv]),
                         ms_per_iter=_stat([x["ms_per_iter"] for x in conv]), max_violation=_stat([x["max_violation"] for x in conv]), cost=_stat([x["cost"] for x in conv]), objective=_stat([x["objective"] for x in conv]),
                         final_time=_stat([x["final_time"] for x in conv]))
json.dump(table, open(f"{OUT}/stats.json", "w"), indent=1)
print("\nsummary (converged runs):")
for name, t in table.items():
    print(f"  {name:22s} {t['converged']:2d}/{t['runs']}  it {t['iterations'][0]:6.1f}+-{t['iterations'][1]:5.1f}  scp {t['scp_time_s'][0]:6.2f}s  ms/it {t['ms_per_iter'][0]:6.1f}  "
          f"viol {t['max_violation'][0]:.2e}  obj {t['objective'][0]:.4f}+-{t['objective'][1]:.4f}")
if args.gamma is None:
    paths = {}
    for run in exp.runs:
        if run.evaluation is None or not run.evaluation.phases: continue
        fin = next(iter(run.evaluation.phases.values())).final
        paths[f"{run.variant}|{run.case}|{int(bool(getattr(run.solve_result, 'converged', False)))}"] = np.asarray(fin.x_nl)[:, :2]
    np.savez(f"{OUT}/trajectories.npz", **paths)
    from plot_dispersed import plot_dispersed
    plot_dispersed(f"{OUT}/trajectories.npz", cfg, f"{OUT}/dispersed_trajectories.pdf")
