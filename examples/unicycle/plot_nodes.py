"""Discrete nodes versus nonlinear re-propagation at two dynamics tolerances, from a guess through the gap between the zones."""
import argparse, copy, os
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from trajopt.trajectory_analyzer import TrajectoryAnalyzer
from trajopt.analysis.experiments import ExperimentCase, ExperimentVariant, run_experiment

ap = argparse.ArgumentParser(); ap.add_argument("config", nargs="?", default="compare_plot.yaml")
ap.add_argument("--eps", nargs=2, type=float, default=[1e-3, 1e-5]); ap.add_argument("--waypoint", nargs=2, type=float, default=[6.0, 4.0])
ap.add_argument("--zoom", nargs=3, type=float, default=[7.0, 5.4, 1.4])
ap.add_argument("--out", default="plots/compare_plot/nodes_vs_propagation.pdf")
args = ap.parse_args()
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})
traj = TrajectoryAnalyzer(args.config); base = copy.deepcopy(traj.config)
base.method.hyperparams.initial_guess.default.waypoint = [args.waypoint[0], args.waypoint[1], 1.4142, 0.7853981633974483, 0.0]
cases = []
for i, e in enumerate(args.eps):
    c = copy.deepcopy(base); c.trajectory.constraints.dynamics.eps.default = e
    cases.append(ExperimentCase(f"eps{i}", c, nominal=(i == 0)))
variants = [ExperimentVariant(v, base.analysis.methods[v]) for v in ("Variant 3a", "Variant 3b")]
exp = run_experiment(cases, variants, evaluate=True, analysis_config=base.analysis)
colors = {"Variant 3a": "C2", "Variant 3b": "C3"}
obstacles = [tuple(o) for o in base.trajectory.params.obstacles]

def node_gap(xo, xn):
    a, b = xn[:-1, :2], xn[1:, :2]; ab = b - a
    t = np.clip(np.einsum("ikj,kj->ik", xo[:, None, :2] - a[None], ab) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    return np.linalg.norm(xo[:, None, :2] - (a[None] + t[..., None] * ab[None]), axis=2).min(axis=1).max()
def clearance(x):
    return min(np.min(np.hypot(x[:, 0] - cx, x[:, 1] - cy) - r) for cx, cy, r in obstacles)
def eps_label(e):
    return rf"$\varepsilon_{{\rm dyn}}=10^{{{int(round(np.log10(e)))}}}$"

fig, axes = plt.subplots(len(args.eps), 2, figsize=(8.0, 3.7 * len(args.eps)))
th = np.linspace(0, 2 * np.pi, 400)
for row in axes:
    for cx, cy, r in obstacles:
        for ax in row: ax.plot(cx + r * np.cos(th), cy + r * np.sin(th), "k--", lw=1.2)
for run in exp.runs:
    if run.evaluation is None or not run.evaluation.phases or not run.solve_result.converged: continue
    i = int(run.case[3:]); row = axes[i]; c = colors[run.variant]
    fin = next(iter(run.evaluation.phases.values())).final
    xo, xn = np.asarray(fin.x_opt), np.asarray(fin.x_nl)
    print(f"{run.variant} eps={args.eps[i]:g}: {run.solve_result.iterations} iterations, node gap {node_gap(xo, xn):.4f} m, "
          f"clearance at nodes {clearance(xo):+.4f} m, along propagation {clearance(xn):+.4f} m")
    for ax in row:
        ax.plot(xn[:, 0], xn[:, 1], "-", color=c, lw=1.5, label=run.variant if ax is row[0] else None)
        ax.plot(xo[:, 0], xo[:, 1], "o", color=c, ms=5 if ax is row[0] else 8, mfc="white", mew=1.5)
cx, cy, hw = args.zoom
for i, row in enumerate(axes):
    row[0].set_xlim(-1, 11); row[0].set_ylim(-1, 11); row[0].set_title("planar path, " + eps_label(args.eps[i]), fontsize=10)
    row[1].set_xlim(cx - hw, cx + hw); row[1].set_ylim(cy - hw, cy + hw); row[1].set_title("second no-fly zone, " + eps_label(args.eps[i]), fontsize=10)
    for ax in row: ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.grid(alpha=0.3)
    row[0].legend(frameon=False, loc="upper left")
fig.tight_layout(); os.makedirs(os.path.dirname(args.out), exist_ok=True); fig.savefig(args.out, bbox_inches="tight"); print("wrote", args.out)
