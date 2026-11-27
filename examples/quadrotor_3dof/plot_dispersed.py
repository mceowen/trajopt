"""Final iterates of the converged dispersed runs, one panel per variant with at least one converged run."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_dispersed(npz, cfg, out, per_row=3):
    data = np.load(npz)
    keys = [k.split("|") for k in data.files]
    variants = []
    for v, _, _ in keys:
        if v not in variants: variants.append(v)
    variants = [v for v in variants if any(ok == "1" for vv, _, ok in keys if vv == v)]
    n = len(variants); rows = int(np.ceil(n / per_row)); cols = min(n, per_row)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.1 * rows), squeeze=False)
    params = cfg.trajectory.get("params", {}); th = np.linspace(0, 2 * np.pi, 200)
    obs = cfg.trajectory.get("constraints", {}).get("obstacle", {})
    circles = [(obs.pos[0], obs.pos[1], float(params.obstacle.radius))] if "pos" in obs and "obstacle" in params else []
    circles += [(c[0], c[1], c[2]) for c in params.get("obstacles", [])]
    for i, v in enumerate(variants):
        ax = axes[i // per_row][i % per_row]
        n_ok = n_all = 0
        for vv, case, ok in keys:
            if vv != v: continue
            n_all += 1; n_ok += ok == "1"
            if ok != "1": continue
            x = data[f"{vv}|{case}|{ok}"]
            ax.plot(x[:, 0], x[:, 1], color="C0", alpha=0.6, lw=1.0)
        for cx, cy, r in circles:
            ax.plot(cx + r * np.cos(th), cy + r * np.sin(th), "k--", lw=1)
        ax.set_title(f"{v} ({n_ok}/{n_all} converged)", fontsize=10)
        ax.set_xlabel("x (m)", fontsize=10); ax.set_ylabel("y (m)", fontsize=10); ax.tick_params(labelsize=9)
        ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlim(-2.5, 12.5); ax.set_ylim(-2.5, 12.5)
    for j in range(n, rows * cols):
        axes[j // per_row][j % per_row].axis("off")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    from trajopt.trajectory_analyzer import TrajectoryAnalyzer
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "dispersed.yaml"
    name = cfg_file.rsplit(".", 1)[0]
    plot_dispersed(f"plots/{name}/trajectories.npz", TrajectoryAnalyzer(cfg_file).config, f"plots/{name}/dispersed_trajectories.pdf")
