"""Dual-step sweep of Algorithm Variant 1 from the nominal guess."""
import json, os, sys
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from trajopt.trajectory_analyzer import TrajectoryAnalyzer
from trajopt.common.tools import deep_merge, recursive_attrdict
from trajopt.analysis.experiments import ExperimentCase, ExperimentVariant, run_experiment

CFG = sys.argv[1] if len(sys.argv) > 1 else "compare_term.yaml"
STEPS = [float(s) for s in (sys.argv[2].split(",") if len(sys.argv) > 2 else "0.5,1,2,5,10,20,50,100".split(","))]
OUT = "plots/" + os.path.splitext(os.path.basename(CFG))[0]
os.makedirs(OUT, exist_ok=True)
traj = TrajectoryAnalyzer(CFG)
cfg = traj.config
base = cfg.analysis.methods["Variant 1"]
variants = [ExperimentVariant(f"Variant 1 (s_dual={s:g})", deep_merge(recursive_attrdict(dict(base)), recursive_attrdict({"hyperparams": {"penalties": {"default": {"dual": {"tr_step": s}}}}}))) for s in STEPS]
exp = run_experiment([ExperimentCase("nominal", cfg, nominal=True)], variants, evaluate=True, analysis_config=cfg.analysis)
summary, paths, hist = {}, {}, {}
for s, run in zip(STEPS, exp.runs):
    r, ev = run.solve_result, run.evaluation
    its = [d for d in next(iter(r.phase_data.values())).iter_data_list if d.get("chk") is not None] if r is not None and getattr(r, "phase_data", None) else []
    row = dict(s_dual=s, converged=bool(getattr(r, "converged", False)), iterations=int(getattr(r, "iterations", 0) or 0), status=str(getattr(r, "status", None)), error=run.error,
               scp_time_s=float(dict(getattr(r, "timings", {}) or {}).get("total_s", np.nan)) if r is not None else np.nan)
    if ev is not None and ev.phases:
        ph = next(iter(ev.phases.values())); it, dg = ph.final, ph.diagnostics
        viol = {k: float(v["maximum"]) for k, v in dg["physical_constraint_violations"].items()}
        last = its[-1] if its else {}
        row.update(objective=float(last.get("cost", np.nan)) - float(last.get("penalty_cost", 0.0) or 0.0), max_violation=max(viol.values()) if viol else None, violations=viol,
                   endpoint_state_error=float(np.abs(np.asarray(dg["endpoint_state_error"])).max()))
        paths[f"{s:g}"] = dict(x_opt=np.asarray(it.x_opt), x_nl=np.asarray(it.x_nl), t_opt=np.asarray(it.t_opt), t_nl=np.asarray(it.t_nl), u_opt=np.asarray(it.u_opt))
    if its:
        hist[f"{s:g}"] = {k: np.array([float(d["chk"].get(k, 0.0)) for d in its]) for k in ("dz", "final_state", "nonconvex_inequality", "dynamics")}
        hist[f"{s:g}"]["objective"] = np.array([float(d.get("cost", np.nan)) - float(d.get("penalty_cost", 0.0) or 0.0) for d in its])
    summary[f"{s:g}"] = row
    print(f"RESULT s_dual={s:g} converged={row['converged']} it={row['iterations']} status={row['status']} obj={row.get('objective')} maxviol={row.get('max_violation')}")
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, default=str)
np.savez(f"{OUT}/trajectories.npz", **{f"{k}|{n}": v for k, d in paths.items() for n, v in d.items()})
np.savez(f"{OUT}/histories.npz", **{f"{k}|{n}": v for k, d in hist.items() for n, v in d.items()})
