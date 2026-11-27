"""Table and figure of the bare powered-descent study: standalone methods and seeded hybrids from the straight-line guess and 20 dispersed guesses."""
import glob, json, os, re, sys
import numpy as np
OUT_TEX = sys.argv[1] if len(sys.argv) > 1 else "plots/pdg_summary.tex"
CASE = re.compile(r"\|(nominal|case_\d{4})")
LABEL = {"MS": "MS (15 nodes, dynamics buffered)", "p": "$p$-term (9 nodes)", "hp": "$hp$ (41 nodes, dynamics buffered)", "hp-term": "$hp$-term (41 nodes, 10$\\times$4)",
         "p>MS": "$p$-term$\\,\\to\\,$MS", "hp-term>MS": "$hp$-term$\\,\\to\\,$MS"}
rows = []
for r in json.load(open("plots/pdg_disp/screen.json")):
    rows.append(dict(r, group=LABEL[r["method"]], order=(0, ["MS", "p", "hp", "hp-term"].index(r["method"]), 0), seed_it=0, finish_it=r["iterations"]))
for f in sorted(glob.glob("plots/pdg_disp_handoff/*/row.json")):
    r = json.load(open(f)); st = r.get("stages", {}); b = r.get("seed_budget")
    rows.append(dict(r, group=LABEL[r["method"]] + (f", seed budget {b}" if b else ", converged seed"), order=(1, ["p>MS", "hp-term>MS"].index(r["method"]), 0 if b is None else 1, b or 0),
                     seed_it=st.get("seed", {}).get("iterations", 0), finish_it=st.get("finish", {}).get("iterations", 0), case=CASE.search(r["tag"]).group(1) if CASE.search(r["tag"]) else "nominal"))
def med(v): v = [x for x in v if x is not None and np.isfinite(x)]; return float(np.median(v)) if v else float("nan")
groups = sorted({r["group"]: r["order"] for r in rows}.items(), key=lambda kv: kv[1]); groups = [g for g, _ in groups]
L = []
for g in groups:
    rs = [r for r in rows if r["group"] == g]; conv = [r for r in rs if r["converged"]]; ok = [r for r in rs if r["pass"]]
    hyb = rs[0]["order"][0] == 1
    it_s = f"{med([r['seed_it'] for r in conv]):.0f}+{med([r['finish_it'] for r in conv]):.0f}" if hyb else f"{med([r['iterations'] for r in conv]):.0f}"
    t = med([r["scp_time_s"] for r in conv]); tf = [r["cost"] for r in ok if r.get("cost") is not None]
    tf_s = f"{min(tf):.2f}--{max(tf):.2f}" if tf else "--"
    print(f"{g:52s} n={len(rs):2d} conv={len(conv):2d} pass={len(ok):2d} it={it_s:8s} t={t:5.1f}s tF(pass)={tf_s}")
    L.append(f"{g} & {len(conv)} & {len(ok)} & {it_s} & {t:.1f} & {tf_s} \\\\")
tex = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
       r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.46\textwidth}rrrrr@{}}", r"\toprule",
       r"\textbf{Method} & \textbf{Conv.} & \textbf{Pass} & \textbf{Iter.} & \textbf{Solve (s)} & $t_F$ \textbf{(pass)} \\", r"\midrule\addlinespace[2pt]", *L, r"\bottomrule", r"\end{tabular}",
       r"\caption{Bare powered-descent problem from the straight-line guess and 20 dispersed guesses (21 guesses): converged and passing runs, median iterations (seed stage$+$final stage for hybrid methods) and median solve time over converged runs, and the range of the minimum-time objective $t_F$ over passing runs. The multiple-shooting final stage of every hybrid method is the standalone shooting configuration.}",
       r"\label{tab:ch6-pdg}", r"\end{table}"]
open(OUT_TEX, "w").write("\n".join(tex) + "\n"); print("wrote", OUT_TEX)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.32 * len(groups) + 1.6), gridspec_kw={"width_ratios": [1.1, 1.0]})
y = np.arange(len(groups))[::-1]
short = [g.replace("-term", "").replace(" (15 nodes, dynamics buffered)", " (15)").replace(" (9 nodes)", " (9)").replace(" (41 nodes, dynamics buffered)", " (41)").replace(" (41 nodes, 10$\\times$4)", " (41)").replace(", converged seed", " (conv.)").replace(", seed budget ", " (b") for g in groups]
short = [x + (")" if " (b" in x and not x.endswith(")") else "") for x in short]
nconv = [sum(r["converged"] for r in rows if r["group"] == g) for g in groups]; npass = [sum(r["pass"] for r in rows if r["group"] == g) for g in groups]
axes[0].barh(y + 0.18, nconv, height=0.34, color="0.75", label="converged"); axes[0].barh(y - 0.18, npass, height=0.34, color="C0", label="passed replay")
axes[0].set_yticks(y); axes[0].set_yticklabels(short); axes[0].set_xlabel("runs of 21 guesses"); axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8, frameon=False); axes[0].grid(axis="x", alpha=0.3)
rng = np.random.RandomState(0)
for yi, g in zip(y, groups):
    for r in [r for r in rows if r["group"] == g and r["converged"] and r.get("cost") is not None]:
        axes[1].plot(r["cost"], yi + rng.uniform(-0.22, 0.22), marker="o", ms=4, mfc=("C0" if r["pass"] else "none"), mec="C0", mew=0.9, ls="none")
axes[1].set_yticks(y); axes[1].set_yticklabels([]); axes[1].set_xlabel(r"$t_F$ of converged runs"); axes[1].grid(axis="x", alpha=0.3)
axes[1].plot([], [], "o", ms=4, color="C0", label="passed"); axes[1].plot([], [], "o", ms=4, mfc="none", mec="C0", label="failed replay"); axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(os.path.splitext(OUT_TEX)[0] + ".pdf"); print("wrote figure")
