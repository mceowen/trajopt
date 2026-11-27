"""Tables and figures of the transcription comparison from the tuning screen and the final runs."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml as _yaml
_m = _yaml.safe_load(open("mission_all.yaml"))
def _lim(key):
    v = _m.get("constraints", {}).get(key, {}) if isinstance(_m.get("constraints"), dict) else {}
    u = v.get("upper") if isinstance(v, dict) else None
    return float(u[0] if isinstance(u, list) else u)
LIMITS = (_lim("max_heat_rate"), _lim("max_dynamic_pressure"), _lim("max_aero_load"))

OUT_TEX = sys.argv[1] if len(sys.argv) > 1 else "plots/final/ch6_tables.tex"
FIG = sys.argv[2] if len(sys.argv) > 2 else "plots/final"
METHODS = ["MS-term", "p-term", "hp-term"]
LABEL = {"MS-term": "MS", "p-term": "$p$", "hp-term": "$hp$"}
COLOR = {"MS-term": "C0", "p-term": "C1", "hp-term": "C2"}
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})

screen = []
for m in METHODS:
    import glob
    for f in sorted(glob.glob(f"plots/matrix_{m}*/screen.json")): screen += json.load(open(f))
sel = json.load(open("plots/matrix/selected.json")); final = json.load(open("plots/final/final.json"))
HANDOFF = ["p-term>MS-term", "hp-term>MS-term"]
hfinal = json.load(open("plots/handoff_final/final.json")) if os.path.exists("plots/handoff_final/final.json") else {}
LABEL.update({"p-term>MS-term": "$p$-term$\\,\\to\\,$MS", "hp-term>MS-term": "$hp$-term$\\,\\to\\,$MS"}); COLOR.update({"p-term>MS-term": "C3", "hp-term>MS-term": "C4"})
FIRST = {"MS-term": "C0", "p-term": "C1", "hp-term": "C2", "p-term>MS-term": "C1", "hp-term>MS-term": "C2"}

def mesh(r):
    return f"{r['N']}" if r["K"] == 1 else f"{r['N']} ({r['K']}$\\times${(r['N'] - 1) // r['K']})"
def ctrl(r):
    return "FOH" if r["method"] == "MS-term" else ("Lagrange" if r["control"] == "lagrange" else f"FOH {r['stride']}")
def sci(x):
    if x is None or not np.isfinite(x): return "--"
    if 0.01 <= abs(x) < 1000: return f"{x:.3g}"
    e = int(np.floor(np.log10(abs(x)))); return f"${x / 10**e:.2f}\\times10^{{{e}}}$"

L = []
L += [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{rlccrrrrrll}", r"\toprule",
      r"\textbf{ID} & \textbf{Method} & \textbf{Mesh $N$} & \textbf{Control} & \textbf{Solve (s)} & \textbf{Iter.} & $\phi_{\mathrm{rep}}(t_F)$ & $\phi_N$ & $R$ & \textbf{Limiting check} & \textbf{Replay} \\", r"\midrule"]
for i, m in enumerate(METHODS, 1):
    r = final[m]
    L.append(f"{i} & {LABEL[m]}-term & {mesh(r)} & {ctrl(r)} & {r['scp_time_median_s']:.2f} & {r['iterations']} & ${r['cost']:.3f}^\\circ$ & ${r['nodal_cost']:.3f}^\\circ$ & {sci(r['R'])} & {r['limiting']} & {'Pass' if r['pass'] else 'Fail'} \\\\")
for i, m in enumerate(HANDOFF, len(METHODS) + 1):
    if m not in hfinal: continue
    r = hfinal[m]; st = r["stages"]
    L.append(f"{i} & {LABEL[m]}-term{'$^{w}$' if r.get('weights') else ''} & {r['seed_N']}$\\to${r['N']} & FOH & {r['scp_time_median_s']:.2f} & {st['seed']['iterations']}+{st['finish']['iterations']} & ${r['cost']:.3f}^\\circ$ & ${r['nodal_cost']:.3f}^\\circ$ & {sci(r['R'])} & {r['limiting']} & {'Pass' if r['pass'] else 'Fail'} \\\\")
L += [r"\bottomrule", r"\end{tabular}}",
      r"\caption{Standalone and hybrid methods on the GHAME entry problem at the configuration selected for each by the tuning screen. Mesh counts are state nodes ($K\times N_k$ for the hp-method), in seed-stage$\to$final-stage order for hybrid methods, whose iterations are listed in the same order. Superscript $w$ marks transfer of terminal penalty weights. Solve time is the median of three serial runs in fresh processes, excluding compilation. $\phi_{\mathrm{rep}}(t_F)$ is the crossrange reached when the optimized control is propagated through the nonlinear dynamics and $\phi_N$ the crossrange at the last node. $R$ is the largest normalized violation of Eq.~\eqref{eq:ch6-replay-accuracy} and the limiting check the one that sets it.}",
      r"\label{tab:ch6-results}", r"\end{table}"]
L += [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lcccccccc}", r"\toprule",
      r"\textbf{Method} & $N$ & $K$ & \textbf{Control} & \textbf{RK4} & $(s_y, s_\nu)$ & \textbf{Screened} & \textbf{Converged} & \textbf{Passing} \\", r"\midrule"]
for m in METHODS:
    r = final[m]; s = sel[m]
    L.append(f"{LABEL[m]}-term & {r['N']} & {r['K'] if r['method'] != 'MS-term' else '--'} & {ctrl(r)} & {'10' if m == 'MS-term' else '--'} & ({r['sz']:g}, {r['su']:g}) & {s['screened']} & {s['converged']} & {s['passing']} \\\\")
for m in HANDOFF:
    if m not in hfinal: continue
    r = hfinal[m]
    L.append(f"{LABEL[m]}-term & {r['seed_N']}$\\to${r['N']} & {r['seed_K']}$\\to$-- & FOH & 10 & ({r['sz']:g}, {r['su']:g}) & {r['screened']} & {r['converged_count']} & {r['passing']} \\\\")
L += [r"\bottomrule", r"\end{tabular}}",
      r"\caption{Settings selected by the tuning screen for each method, and the number of screened configurations that converged and that passed the acceptance test. RK4 is the number of substeps per shooting interval; for the hybrid methods the mesh, control, and steps are those of the final stage. All other settings are those of Table~\ref{tab:ch6-setup}.}",
      r"\label{tab:ch6-settings}", r"\end{table}"]
os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True); open(OUT_TEX, "w").write("\n".join(L) + "\n"); print("wrote", OUT_TEX)

# ---- screen figure: replayed crossrange and R against node count ----
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
for m in METHODS:
    rs = [r for r in screen if r["method"] == m and r.get("R") is not None]
    for conv in (True, False):
        sub = [r for r in rs if r["converged"] == conv]
        if not sub: continue
        for st in (1, 2):
            ss = [r for r in sub if r["control"] == "foh" and (r["stride"] == st or m == "MS-term")]
            if not ss or (m == "MS-term" and st == 2): continue
            N = [r["N"] for r in ss]; c = [r["cost"] for r in ss]; R = [r["R"] for r in ss]
            kw = dict(color=COLOR[m], marker=("o" if st == 1 else "s"), ls="none", ms=5, mfc=(COLOR[m] if conv else "white"), alpha=0.9)
            lab = f"{LABEL[m]}-term" + ("" if m == "MS-term" else f", FOH {st}") + ("" if conv else " (not converged)")
            axes[0].plot(N, c, label=lab, **kw); axes[1].plot(N, R, **kw)
axes[0].set_xlabel("state nodes $N$"); axes[0].set_ylabel(r"$\phi_{\rm rep}(t_F)$ (deg)"); axes[0].grid(alpha=0.3)
axes[1].set_xlabel("state nodes $N$"); axes[1].set_ylabel("largest normalized violation $R$"); axes[1].set_yscale("log"); axes[1].axhline(1.0, color="k", ls=":", lw=1); axes[1].grid(alpha=0.3, which="both")
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(); fig.savefig(f"{FIG}/screen.pdf", bbox_inches="tight"); plt.close(fig); print("wrote screen.pdf")

# ---- histories of the selected runs ----
lim = None
fig, axes = plt.subplots(3, 3, figsize=(8.5, 8.0))
def runs_for_plots():
    out = []
    for m in METHODS:
        r = final[m]; out.append((m, np.load(os.path.join(r["rep_dir"], r["tag"].replace("|", "_") + ".npz")), r))
    for m in HANDOFF:
        if m in hfinal: out.append((m, np.load(os.path.join(hfinal[m]["rep_dir"], "replay.npz")), hfinal[m]))
    return out
for m, d, r in runs_for_plots():
    t, X, U = d["dense_t"], d["dense_x"], d["dense_u"]; tn, Xn, Un = d["nodes_t"], d["nodes_x"], d["nodes_u"]
    rows = [((X[:, 0] - 6378137.0) / 1e3, (Xn[:, 0] - 6378137.0) / 1e3, "altitude (km)"), (X[:, 3] / 1e3, Xn[:, 3] / 1e3, "velocity (km/s)"), (X[:, 4], Xn[:, 4], "flight-path angle (deg)"),
            (X[:, 5], Xn[:, 5], "heading (deg)"), (U[:, 0], Un[:, 0], "bank angle (deg)"), (U[:, 1], Un[:, 1], "angle of attack (deg)"),
            (d["path_heat_rate"], None, r"heat rate (W/cm$^2$)"), (d["path_dynamic_pressure"], None, "dynamic pressure (kPa)"), (d["path_load"], None, "load ($g$)")]
    for ax, (y, yn, lab) in zip(axes.ravel(), rows):
        ax.plot(t, y, color=COLOR[m], lw=1.3, label=f"{LABEL[m]}-term")
        if yn is not None: ax.plot(tn, yn, "o", color=COLOR[m], ms=2.5, mfc="white", mew=0.8)
        ax.set_ylabel(lab); ax.grid(alpha=0.3)
for ax, lim in zip(axes.ravel()[6:], LIMITS): ax.axhline(lim, color="k", ls="--", lw=1)
for ax, (lo, hi) in zip(axes.ravel()[4:6], ((-80, 80), (10, 20))):
    ax.axhline(lo, color="k", ls=":", lw=0.8); ax.axhline(hi, color="k", ls=":", lw=0.8)
for ax in axes[-1]: ax.set_xlabel("time (s)")
h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(); fig.savefig(f"{FIG}/histories.pdf", bbox_inches="tight"); plt.close(fig); print("wrote histories.pdf")

# ---- ground tracks ----
fig, ax = plt.subplots(figsize=(4.2, 3.4))
for m, d, r in runs_for_plots():
    ax.plot(d["dense_x"][:, 1], d["dense_x"][:, 2], color=COLOR[m], lw=1.3, label=f"{LABEL[m]}-term"); ax.plot(d["nodes_x"][:, 1], d["nodes_x"][:, 2], "o", color=COLOR[m], ms=2.5, mfc="white", mew=0.8)
ax.set_xlabel("longitude (deg)"); ax.set_ylabel("latitude (deg)"); ax.grid(alpha=0.3); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{FIG}/ground_track.pdf", bbox_inches="tight"); plt.close(fig); print("wrote ground_track.pdf")

# ---- convergence histories of the selected runs ----
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
for m in METHODS + [h for h in HANDOFF if h in hfinal]:
    r = final[m] if m in final else hfinal[m]
    f = os.path.join(r["rep_dir"], (r["tag"].replace("|", "_") + "_iters.npz") if m in final else "iters.npz")
    if not os.path.exists(f): continue
    d = np.load(f); k = np.arange(1, len(d["dz"]) + 1)
    axes[0].plot(k, np.maximum(d["dz"], 1e-6), color=COLOR[m], lw=1.3, label=f"{LABEL[m]}-term, state step")
    axes[0].plot(k, np.maximum(np.maximum(d["final_state"], d["nonconvex_inequality"]), 1e-6), color=COLOR[m], lw=1.0, ls="--", label=f"{LABEL[m]}-term, buffers")
    axes[1].plot(k, 90.0 * d["objective"], color=COLOR[m], lw=1.3, label=f"{LABEL[m]}-term")
axes[0].axhline(1.0, color="k", ls=":", lw=1); axes[0].set_yscale("log"); axes[0].set_ylabel("normalized state step and buffer"); axes[0].set_xlabel("SCP iteration"); axes[0].grid(alpha=0.3, which="both")
axes[1].set_ylabel(r"$\phi_N$ (deg)"); axes[1].set_xlabel("SCP iteration"); axes[1].grid(alpha=0.3)
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(); fig.savefig(f"{FIG}/convergence.pdf", bbox_inches="tight"); plt.close(fig); print("wrote convergence.pdf")

# ---- solve time against replayed crossrange ----
fig, ax = plt.subplots(figsize=(4.6, 3.4))
for i, m in enumerate(METHODS + HANDOFF, 1):
    r = final.get(m) or hfinal.get(m)
    if r is None: continue
    ax.plot(r["scp_time_median_s"], r["cost"], ls="none", ms=7, color=FIRST[m],
            marker=("o" if r["pass"] else "x"), mfc=(FIRST[m] if r["pass"] else "none"), mew=1.6)
    ax.annotate(str(i), (r["scp_time_median_s"], r["cost"]), textcoords="offset points", xytext=(5, 4), fontsize=9)
ax.set_xscale("log"); ax.set_xlabel("solve time (s)"); ax.set_ylabel(r"$\phi_{\mathrm{rep}}(t_F)$ (deg)"); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(f"{FIG}/runtime_cost.pdf", bbox_inches="tight"); plt.close(fig); print("wrote runtime_cost.pdf")
