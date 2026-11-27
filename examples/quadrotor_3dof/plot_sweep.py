"""Dual-step sweep of Variant 1: converged trajectories and convergence histories."""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("sweep_dir", nargs="?", default="plots/sweep_term"); ap.add_argument("--out", default=None)
ap.add_argument("--config", default="compare_term.yaml")
args = ap.parse_args(); out = args.out or os.path.join(args.sweep_dir, "sweep.pdf")
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})
summary = json.load(open(os.path.join(args.sweep_dir, "summary.json")))
paths = np.load(os.path.join(args.sweep_dir, "trajectories.npz")); hist = np.load(os.path.join(args.sweep_dir, "histories.npz"))
from trajopt.trajectory_analyzer import TrajectoryAnalyzer
cfg = TrajectoryAnalyzer(args.config).config
params = cfg.trajectory.get("params", {}); th = np.linspace(0, 2 * np.pi, 300)
obs = cfg.trajectory.get("constraints", {}).get("obstacle", {})
circles = [(obs.pos[0], obs.pos[1], float(params.obstacle.radius))] if "pos" in obs and "obstacle" in params else []
circles += [(c[0], c[1], c[2]) for c in params.get("obstacles", [])]
fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2), gridspec_kw=dict(width_ratios=[1.0, 1.1, 1.1]))
for cx, cy, r in circles: axes[0].plot(cx + r * np.cos(th), cy + r * np.sin(th), "k--", lw=1.2)
steps = sorted(summary, key=float); j = 0
for k in steps:
    row = summary[k]
    if not row["converged"] or f"{k}|x_opt" not in paths: continue
    xo, xn = paths[f"{k}|x_opt"], paths[f"{k}|x_nl"]; c = f"C{j}"; j += 1
    axes[0].plot(xn[:, 0], xn[:, 1], "-", color=c, lw=1.4, label=rf"$s_{{\rm dual}}={k}$ ({row['iterations']} it.)")
    axes[0].plot(xo[:, 0], xo[:, 1], "o", color=c, ms=3.5, mfc="white", mew=1.0)
    h = {n: hist[f"{k}|{n}"] for n in ("dz", "final_state", "nonconvex_inequality")}; it = np.arange(1, len(h["dz"]) + 1)
    axes[1].plot(it, np.maximum(h["dz"], 1e-4), color=c, lw=1.2)
    axes[1].plot(it, np.maximum(np.maximum(h["final_state"], h["nonconvex_inequality"]), 1e-4), color=c, lw=1.0, ls="--")
axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)"); axes[0].set_aspect("equal"); axes[0].grid(alpha=0.3); axes[0].legend(frameon=False, fontsize=8)
axes[0].set_title("converged solutions (markers: nodes; lines: re-propagation)", fontsize=9)
axes[1].axhline(1.0, color="k", ls=":", lw=1); axes[1].set_yscale("log"); axes[1].set_xlabel("SCP iteration"); axes[1].set_ylabel("ratio to tolerance"); axes[1].set_title("state step (solid), buffer violation (dashed)", fontsize=9); axes[1].grid(alpha=0.3, which="both")
its = [summary[k]["iterations"] if summary[k]["converged"] else np.nan for k in steps]; s = [float(k) for k in steps]
axes[2].plot(s, its, "o-", color="k", ms=5)
for k, sv in zip(steps, s):
    if not summary[k]["converged"]: axes[2].plot(sv, 2000, "x", color="r", ms=7, mew=1.5)
axes[2].set_xscale("log"); axes[2].set_xlabel(r"dual step $s_{\rm dual}$"); axes[2].set_ylabel("iterations to converge"); axes[2].grid(alpha=0.3, which="both"); axes[2].set_yscale("log"); axes[2].set_ylim(50, 3000)
axes[2].set_title("crosses: not converged within 2000 iterations", fontsize=9)
fig.tight_layout(); os.makedirs(os.path.dirname(out), exist_ok=True); fig.savefig(out, bbox_inches="tight"); print("wrote", out)
