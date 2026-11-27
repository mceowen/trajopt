import json, os
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trajopt.trajectory_analyzer import TrajectoryAnalyzer

OUT = "plots/compare"
traj = TrajectoryAnalyzer("compare.yaml")
exp = traj.analyze()
traj.plot(save_dir=OUT)

summary, buffers = {}, {}
for run in exp.runs:
    r = run.solve_result
    it = next(iter(r.phase_data.values())).iter_data_list[-1]
    vb_all = it["vb"]
    vb = np.asarray(vb_all["dynamics_state"]) if "dynamics_state" in vb_all else np.asarray(vb_all["dynamics"])[:, :14]
    def dyn_weights(iterate):
        if "W_p" in iterate and "dynamics_state" in iterate["W_p"]:
            return np.maximum(np.asarray(iterate["W_p"]["dynamics_state"]), np.asarray(iterate["W_m"]["dynamics_state"]))
        return np.asarray(iterate["W"]["dynamics"])[:, :14]
    iterates = next(iter(r.phase_data.values())).iter_data_list[1:]
    W_dyn = dyn_weights(it)
    W_dyn_run_max = float(max(np.max(dyn_weights(x)) for x in iterates))
    buffers[run.variant] = vb
    summary[run.variant] = dict(converged=bool(r.converged), iterations=int(r.iterations),
                                t_final=float(it["t_final"]), cost=float(it["cost"]),
                                wall_s=float(run.wall_s),
                                max_W_dyn_final=float(np.max(W_dyn)), max_W_dyn_run=W_dyn_run_max,
                                buffer_l1_per_interval=np.abs(vb).sum(axis=1).tolist())
    print(f"RESULT variant={run.variant} converged={r.converged} iterations={r.iterations} "
          f"tF={it['t_final']:.4f} cost={it['cost']:.4f} maxW_run={W_dyn_run_max:.3e} maxW_final={np.max(W_dyn):.3e} wall={run.wall_s:.1f}s")
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)

names = list(buffers)
fig = plt.figure(figsize=(10, 6.2))
gs = fig.add_gridspec(2, len(names), height_ratios=[1.0, 1.15])
ax = fig.add_subplot(gs[0, :])
for name in names:
    ax.semilogy(np.arange(1, buffers[name].shape[0] + 1), np.abs(buffers[name]).sum(axis=1) + 1e-16, "o-", ms=3.5, label=name)
ax.set_xlabel("interval $k$"); ax.set_ylabel(r"$\|\mathbf{p}_{\mathrm{dyn},k}\|_1$ at convergence"); ax.grid(alpha=0.3); ax.legend()
vmax = max(np.abs(b).max() for b in buffers.values()) + 1e-16
for j, name in enumerate(names):
    axh = fig.add_subplot(gs[1, j])
    im = axh.imshow(np.abs(buffers[name]).T, aspect="auto", origin="lower", cmap="viridis",
                    norm=matplotlib.colors.LogNorm(vmin=max(vmax * 1e-8, 1e-16), vmax=vmax))
    axh.set_title(name); axh.set_xlabel("interval $k$")
    if j == 0: axh.set_ylabel("state component")
fig.colorbar(im, ax=fig.axes[1:], shrink=0.9, label=r"$|p_{\mathrm{dyn},k,i}|$")
fig.savefig(f"{OUT}/dynamics_buffer_activation.pdf", bbox_inches="tight"); plt.close(fig)
print("wrote", f"{OUT}/dynamics_buffer_activation.pdf")
