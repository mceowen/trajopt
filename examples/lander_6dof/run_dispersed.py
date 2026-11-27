import json, os
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trajopt.trajectory_analyzer import TrajectoryAnalyzer
from trajopt.analysis.experiments import ExperimentVariant, monte_carlo_cases, run_experiment
from trajopt.analysis.statistics import analyze_quality_metrics

OUT = "plots/dispersed"
os.makedirs(OUT, exist_ok=True)

traj = TrajectoryAnalyzer("dispersed.yaml")
cfg = traj.config
cfg["variations"] = {
    "seed": 0,
    "num": 20,
    "samples": {
        "method.hyperparams.initial_guess.default.x_start": {
            "type": "uniform",
            "lb": [0, -1.0, -1.0, -1.0, -0.5, -0.5, -0.5, 0, 0, 0, 0, 0, 0, 0],
            "ub": [0,  1.0,  1.0,  1.0,  0.5,  0.5,  0.5, 0, 0, 0, 0, 0, 0, 0]},
        "method.hyperparams.initial_guess.default.u_start": {
            "type": "uniform", "lb": [-0.5, -0.5, -0.5], "ub": [0.5, 0.5, 0.5]},
        "method.hyperparams.initial_guess.default.u_stop": {
            "type": "uniform", "lb": [-0.5, -0.5, -0.5], "ub": [0.5, 0.5, 0.5]},
    },
}
cases = monte_carlo_cases(cfg)
variants = [ExperimentVariant(name, overrides or {}) for name, overrides in cfg.analysis.methods.items()]
print(f"{len(cases)} cases x {len(variants)} variants")
exp = run_experiment(cases, variants, evaluate=True, analysis_config=cfg.analysis)

rows = []
for run in exp.runs:
    r = run.solve_result
    ev = run.evaluation
    final_time = None
    if ev is not None:
        it = next(iter(ev.phases.values())).iterates[-1]
        final_time = float(np.asarray(it.t_opt)[-1])
    rows.append(dict(case=run.case, variant=run.variant, converged=bool(getattr(r, "converged", False)),
                     iterations=int(getattr(r, "iterations", 0) or 0), final_time=final_time,
                     wall_s=run.wall_s, error=run.error))
json.dump(rows, open(f"{OUT}/runs.json", "w"), indent=1)

def _stat(vals): 
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))
table = {}
for v in variants:
    sel = [r for r in rows if r["variant"] == v.name]
    conv = [r for r in sel if r["converged"]]
    table[v.name] = dict(runs=len(sel), converged=len(conv), converged_pct=100.0 * len(conv) / max(len(sel), 1),
                         iterations=_stat([r["iterations"] for r in conv]),
                         final_time=_stat([r["final_time"] for r in conv]),
                         wall_s=_stat([r["wall_s"] for r in conv]))
json.dump(table, open(f"{OUT}/stats.json", "w"), indent=1)
with open(f"{OUT}/stats_table.tex", "w") as f:
    f.write("% method & runs & converged & iterations (mean +- std) & final time (mean +- std) & wall time [s]\n")
    for name, t in table.items():
        f.write(f"{name} & {t['runs']} & {t['converged']} ({t['converged_pct']:.0f}\\%) & "
                f"{t['iterations'][0]:.1f} $\\pm$ {t['iterations'][1]:.1f} & {t['final_time'][0]:.3f} $\\pm$ {t['final_time'][1]:.3f} & "
                f"{t['wall_s'][0]:.1f} \\\\\n")
try:
    stats = analyze_quality_metrics(exp, filename=f"{OUT}/stats.tex")
    for name, tbl in stats["table_cell"].items():
        print(f"\n=== {name} ===\n{tbl}")
except ImportError as exc:
    print(f"(library statistics skipped: {exc})")

fig, axes = plt.subplots(1, len(variants), figsize=(4.2 * len(variants), 4), sharey=True)
axes = np.atleast_1d(axes)
for ax, variant in zip(axes, variants):
    n_ok = 0
    for run in exp.runs:
        if run.variant != variant.name or run.evaluation is None:
            continue
        it = next(iter(run.evaluation.phases.values())).iterates[-1]
        x = np.asarray(it.x_nl)
        ok = bool(getattr(run.solve_result, "converged", False))
        n_ok += ok
        ax.plot(x[:, 2], x[:, 1], color="C0" if ok else "C3", alpha=0.6 if ok else 0.9, lw=1.0)
    ax.set_title(f"{variant.name} ({n_ok} converged)")
    ax.set_xlabel("Y"); ax.grid(alpha=0.3)
axes[0].set_ylabel("Altitude")
fig.tight_layout(); fig.savefig(f"{OUT}/dispersed_trajectories.pdf"); plt.close(fig)
print("\nsummary:")
for v in variants:
    sel = [r for r in rows if r["variant"] == v.name]
    conv = [r for r in sel if r["converged"]]
    print(f"  {v.name:18s} converged {len(conv)}/{len(sel)}  mean iters {np.mean([r['iterations'] for r in conv]) if conv else float('nan'):.1f}"
          f"  mean tF {np.mean([r['final_time'] for r in conv if r['final_time'] is not None]) if conv else float('nan'):.3f}")
