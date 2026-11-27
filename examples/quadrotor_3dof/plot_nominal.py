"""Converged nominal solutions: planar path with the no-fly zones and the control history, nodes and nonlinear re-propagation."""
import argparse, os
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from trajopt.trajectory_analyzer import TrajectoryAnalyzer

STYLE = {"Variant 1": "C0", "Variant 2a": "C1", "Variant 3a": "C2", "Variant 2b": "C4", "Variant 3b": "C3"}
ap = argparse.ArgumentParser(); ap.add_argument("config", nargs="?", default="compare_plot.yaml"); ap.add_argument("--out", default=None)
args = ap.parse_args()
out = args.out or f"plots/{os.path.splitext(os.path.basename(args.config))[0]}/nominal.pdf"
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})
traj = TrajectoryAnalyzer(args.config); exp = traj.analyze(); cfg = traj.config
params = cfg.trajectory.get("params", {}); th = np.linspace(0, 2 * np.pi, 300)
obs = cfg.trajectory.get("constraints", {}).get("obstacle", {})
circles = [(obs.pos[0], obs.pos[1], float(params.obstacle.radius))] if "pos" in obs and "obstacle" in params else []
circles += [(c[0], c[1], c[2]) for c in params.get("obstacles", [])]
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), gridspec_kw=dict(width_ratios=[1.0, 1.25]))
for cx, cy, r in circles:
    axes[0].plot(cx + r * np.cos(th), cy + r * np.sin(th), "k--", lw=1.2)
quad = None
for run in exp.runs:
    r = run.solve_result
    if r is None or not getattr(r, "converged", False) or run.evaluation is None or not run.evaluation.phases: continue
    fin = next(iter(run.evaluation.phases.values())).final
    xo, xn, uo, t = np.asarray(fin.x_opt), np.asarray(fin.x_nl), np.asarray(fin.u_opt), np.asarray(fin.t_opt)
    c = STYLE.get(run.variant, "k")
    axes[0].plot(xn[:, 0], xn[:, 1], "-", color=c, lw=1.5, label=run.variant)
    axes[0].plot(xo[:, 0], xo[:, 1], "o", color=c, ms=4, mfc="white", mew=1.2)
    quad = uo.shape[1] == 3
    if quad:
        axes[1].plot(t, np.linalg.norm(uo, axis=1), "-", color=c, lw=1.5, label=run.variant)
    else:
        axes[1].plot(t, uo[:, 0], "-", color=c, lw=1.5, label=run.variant)
        axes[1].plot(t, uo[:, 1], "--", color=c, lw=1.2)
axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)"); axes[0].set_aspect("equal"); axes[0].grid(alpha=0.3); axes[0].legend(frameon=False)
axes[1].set_xlabel("time (s)"); axes[1].grid(alpha=0.3)
if quad:
    v = cfg.trajectory.params.vehicle
    for lim in (float(v.T_min), float(v.T_max)): axes[1].axhline(lim, color="k", ls=":", lw=1)
    axes[1].set_ylabel(r"$\|T\|$ (N)")
else:
    axes[1].set_ylabel(r"$a$ (m s$^{-2}$, solid),  $\Omega$ (rad s$^{-2}$, dashed)")
fig.tight_layout(); os.makedirs(os.path.dirname(out), exist_ok=True); fig.savefig(out, bbox_inches="tight"); print("wrote", out)
