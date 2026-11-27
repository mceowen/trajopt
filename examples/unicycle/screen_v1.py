"""Screen of Algorithm Variant 1 over the dual step and the primal step from the nominal guess."""
import argparse, itertools, json, os
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from trajopt.trajectory_analyzer import TrajectoryAnalyzer
from trajopt.common.tools import deep_merge, recursive_attrdict
from trajopt.analysis.experiments import ExperimentCase, ExperimentVariant, run_experiment

ap = argparse.ArgumentParser()
ap.add_argument("config", nargs="?", default="sweep_std.yaml")
ap.add_argument("--steps", nargs="+", type=float, default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
ap.add_argument("--sz", nargs="+", type=float, default=[0.5, 0.2, 0.1, 0.05])
ap.add_argument("--eps-scale", type=float, default=None, help="multiply the state-step tolerance")
ap.add_argument("--out", default=None)
args = ap.parse_args()
OUT = args.out or "plots/screen_v1_" + os.path.splitext(os.path.basename(args.config))[0].replace("sweep_", "")
os.makedirs(OUT, exist_ok=True)
cfg = TrajectoryAnalyzer(args.config).config
base = cfg.analysis.methods["Variant 1"]
grid = list(itertools.product(args.steps, args.sz))

def overrides(s, sz):
    o = {"hyperparams": {"penalties": {"default": {"dual": {"tr_step": s}}}, "trust_region": {"tr_step": {"z": {"default": sz}}}}}
    if args.eps_scale is not None:
        o["hyperparams"]["convergence"] = {"eps_state": float(cfg.method.hyperparams.convergence.eps_state) * args.eps_scale}
    return o

variants = [ExperimentVariant(f"Variant 1 (s_dual={s:g}, s_z={sz:g})", deep_merge(recursive_attrdict(dict(base)), recursive_attrdict(overrides(s, sz)))) for s, sz in grid]
exp = run_experiment([ExperimentCase("nominal", cfg, nominal=True)], variants, evaluate=True, analysis_config=cfg.analysis)
summary, hist = {}, {}
for (s, sz), run in zip(grid, exp.runs):
    r, ev = run.solve_result, run.evaluation
    its = [d for d in next(iter(r.phase_data.values())).iter_data_list if d.get("chk") is not None] if r is not None and getattr(r, "phase_data", None) else []
    key = f"{s:g}|{sz:g}"
    row = dict(s_dual=s, s_z=sz, converged=bool(getattr(r, "converged", False)), iterations=int(getattr(r, "iterations", 0) or 0), status=str(getattr(r, "status", None)), error=run.error)
    if its:
        hist[key] = {k: np.array([float(d["chk"].get(k, 0.0)) for d in its]) for k in ("dz", "final_state", "nonconvex_inequality", "dynamics")}
        row.update(final_dyn=float(hist[key]["dynamics"][-1]), min_dyn=float(hist[key]["dynamics"].min()), final_dz=float(hist[key]["dz"][-1]))
    if ev is not None and ev.phases:
        dg = next(iter(ev.phases.values())).diagnostics
        viol = {k: float(v["maximum"]) for k, v in dg["physical_constraint_violations"].items()}
        last = its[-1] if its else {}
        row.update(objective=float(last.get("cost", np.nan)) - float(last.get("penalty_cost", 0.0) or 0.0), max_violation=max(viol.values()) if viol else None)
    summary[key] = row
    print(f"RESULT s_dual={s:g} s_z={sz:g} converged={row['converged']} it={row['iterations']} status={row['status']} obj={row.get('objective')} maxviol={row.get('max_violation')} final_dyn={row.get('final_dyn')} min_dyn={row.get('min_dyn')}", flush=True)
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, default=str)
np.savez(f"{OUT}/histories.npz", **{f"{k}|{n}": v for k, d in hist.items() for n, v in d.items()})
