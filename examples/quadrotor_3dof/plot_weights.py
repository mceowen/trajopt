"""Quadratic and linear penalty weights of every buffered constraint and node for selected variants."""
import argparse, os
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from trajopt.trajectory_analyzer import TrajectoryAnalyzer

ap = argparse.ArgumentParser(); ap.add_argument("config", nargs="?", default="compare_plot.yaml")
ap.add_argument("--variants", nargs="*", default=["Variant 3a", "Variant 3b"]); ap.add_argument("--out", default=None)
args = ap.parse_args()
out = args.out or f"plots/{os.path.splitext(os.path.basename(args.config))[0]}/weights.pdf"
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})
traj = TrajectoryAnalyzer(args.config)
traj.config.analysis.methods = {v: traj.config.analysis.methods[v] for v in args.variants}
exp = traj.analyze()
fig, axes = plt.subplots(len(args.variants), 2, figsize=(8.0, 2.9 * len(args.variants)), squeeze=False)
colors = {}
for i, run in enumerate(exp.runs):
    its = [d for d in next(iter(run.solve_result.phase_data.values())).iter_data_list if d.get("chk") is not None]
    k = np.arange(1, len(its) + 1)
    for j, key in enumerate(("W", "dual")):
        ax = axes[i][j]
        for name in its[-1][key]:
            c = colors.setdefault(name, f"C{len(colors)}")
            series = np.array([np.asarray(d[key][name]).reshape(-1) for d in its])
            if series.size == 0: continue
            ax.plot(k, np.abs(series), color=c, lw=0.6, alpha=0.6)
            ax.plot([], [], color=c, lw=1.5, label=name.replace("_", " "))
        ax.set_yscale("log"); ax.grid(alpha=0.3, which="both"); ax.set_xlabel("SCP iteration")
        ax.set_ylabel(r"$\omega$ (quadratic)" if key == "W" else r"$|\bar\lambda|,\ \bar\mu$ (linear)")
        ax.set_title(run.variant, fontsize=10)
h, l = axes[0][1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(); os.makedirs(os.path.dirname(out), exist_ok=True); fig.savefig(out, bbox_inches="tight"); print("wrote", out)
