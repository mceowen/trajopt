"""Nominal comparison of the algorithm variants."""
import json, os, sys, time
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
from trajopt.trajectory_analyzer import TrajectoryAnalyzer

CFG = sys.argv[1] if len(sys.argv) > 1 else "compare.yaml"
OUT = "plots/" + os.path.splitext(os.path.basename(CFG))[0]
os.makedirs(OUT, exist_ok=True)
traj = TrajectoryAnalyzer(CFG)
exp = traj.analyze()
traj.plot(exp, save_dir=OUT)
summary = {}
for run in exp.runs:
    r, ev = run.solve_result, run.evaluation
    if r is None or ev is None or not ev.phases:
        summary[run.variant] = dict(converged=False, status=str(getattr(r, "status", None)), iterations=int(getattr(r, "iterations", 0) or 0), error=run.error)
        print(f"RESULT variant={run.variant} FAILED status={summary[run.variant]['status']} error={run.error}"); continue
    ph = next(iter(ev.phases.values())); it, dg = ph.final, ph.diagnostics
    viol = {k: float(v["maximum"]) for k, v in dg["physical_constraint_violations"].items()}
    tm = dict(r.timings)
    last = next(iter(r.phase_data.values())).iter_data_list[-1]
    def _maxabs(block):
        return {k: float(np.max(np.abs(np.asarray(v)))) for k, v in dict(block or {}).items() if np.size(v)}
    weights = dict(W=_maxabs(last.get("W")), dual=_maxabs(last.get("dual")))
    summary[run.variant] = dict(converged=bool(r.converged), status=str(r.status), iterations=int(r.iterations),
                                scp_time_s=float(tm.get("total_s", 0.0)), ms_per_iter=1000 * float(tm.get("total_s", 0.0)) / max(int(r.iterations), 1),
                                wall_s=float(run.wall_s), final_time=float(np.asarray(it.t_opt)[-1]), cost=float(last.get("cost", np.nan)),
                                objective=float(last.get("cost", np.nan)) - float(last.get("penalty_cost", 0.0) or 0.0),
                                max_violation=max(viol.values()) if viol else 0.0, violations=viol,
                                endpoint_state_error=np.abs(np.asarray(dg["endpoint_state_error"])).max().item(), error=run.error,
                                max_W_final=weights["W"], max_dual_final=weights["dual"])
    s = summary[run.variant]
    print(f"RESULT variant={run.variant} converged={s['converged']} it={s['iterations']} scp={s['scp_time_s']:.2f}s ms/it={s['ms_per_iter']:.0f} tF={s['final_time']:.3f} obj={s['objective']:.4f} cost={s['cost']:.4f} maxviol={s['max_violation']:.2e} dx_end={s['endpoint_state_error']:.2e}")
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
