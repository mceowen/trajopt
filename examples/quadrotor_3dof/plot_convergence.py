"""Convergence histories of the converged variants: largest buffer violation relative to its tolerance and objective."""
import argparse, os
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from trajopt.trajectory_analyzer import TrajectoryAnalyzer

STYLE = {"Variant 1": dict(color="C0", ls="-"), "Variant 2a": dict(color="C1", ls="--"), "Variant 3a": dict(color="C2", ls="--"),
         "Variant 2b": dict(color="C4", ls="-"), "Variant 3b": dict(color="C3", ls="-")}
ap = argparse.ArgumentParser(); ap.add_argument("config", nargs="?", default="compare.yaml")
ap.add_argument("--out", default=None); ap.add_argument("--all", action="store_true")
args = ap.parse_args()
out = args.out or f"plots/{os.path.splitext(os.path.basename(args.config))[0]}/convergence_all.pdf"
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})
traj = TrajectoryAnalyzer(args.config); exp = traj.analyze()
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
n_max = 1; finals = []
for run in exp.runs:
    r = run.solve_result
    if r is None or not getattr(r, "phase_data", None) or (not args.all and not getattr(r, "converged", False)): continue
    its = [d for d in next(iter(r.phase_data.values())).iter_data_list if d.get("chk") is not None]
    if not its: continue
    k = np.arange(1, len(its) + 1); n_max = max(n_max, len(its))
    viol = [max(float(d["chk"].get(key, 0.0)) for key in ("dynamics", "final_state", "nonconvex_inequality", "nonconvex_equality")) for d in its]
    obj = [float(d.get("cost", np.nan)) - float(d.get("penalty_cost", 0.0) or 0.0) for d in its]
    st = STYLE.get(run.variant, dict(color="k", ls=":"))
    axes[0].plot(k, np.maximum(viol, 1e-6), lw=1.5, label=run.variant, **st)
    axes[1].plot(k, obj, lw=1.5, label=run.variant, **st); finals.append(obj[-1])
axes[0].axhline(1.0, color="k", ls=":", lw=1); axes[0].set_yscale("log"); axes[0].set_ylabel(r"$\max\,|{\rm buffer}|/\varepsilon$")
axes[1].set_ylabel("objective")
if finals: axes[1].set_ylim(0, 1.6 * max(finals))
for ax in axes:
    ax.set_xlabel("SCP iteration"); ax.grid(alpha=0.3, which="both"); ax.set_xlim(1, n_max)
    if n_max > 300: ax.set_xscale("log")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(); os.makedirs(os.path.dirname(out), exist_ok=True); fig.savefig(out, bbox_inches="tight"); print("wrote", out)
