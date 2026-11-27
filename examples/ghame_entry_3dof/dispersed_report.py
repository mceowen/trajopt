"""Robustness study over dispersed initial guesses: standalone transcriptions and hybrid methods from the same 21 guesses.
Reads plots/disp_* (matrix.py --dispersed) and plots/disp_handoff_* (handoff.py --guess) and writes a LaTeX table."""
import glob, json, os, re, sys
import numpy as np

OUT_TEX = sys.argv[1] if len(sys.argv) > 1 else "plots/disp_summary.tex"
CASE = re.compile(r"\|(nominal|case_\d{4})")

def load():
    rows = []
    for f in sorted(glob.glob("plots/disp_*/screen.json")):
        for r in json.load(open(f)):
            key = f"{r['method']} N{r['N']}" + (f" ({r['K']}$\\times${(r['N']-1)//r['K']})" if r["method"] != "MS-term" and r["K"] > 1 else "") + f", $s_y={r['sz']:g}$"
            rows.append(dict(r, group=key, kind="standalone", seed_it=0, finish_it=r["iterations"]))
    for f in sorted(glob.glob("plots/disp_handoff_*/*/row.json")):
        r = json.load(open(f)); st = r.get("stages", {})
        b = re.search(r"\|b(\d+)", r["tag"]); s = re.search(r"\|s(\d+)", r["tag"])
        key = f"{r['method']} {r['seed_N']}$\\to${r['N']}" + (f", seed budget {b.group(1)}" if b else ", seed converged")
        rows.append(dict(r, group=key, kind="hybrid", seed_it=st.get("seed", {}).get("iterations", 0), finish_it=st.get("finish", {}).get("iterations", 0),
                         case=(CASE.search(r["tag"]).group(1) if CASE.search(r["tag"]) else "nominal")))
    return rows

def med(v): v = [x for x in v if x is not None and np.isfinite(x)]; return float(np.median(v)) if v else float("nan")

def order(g):
    kind = 1 if "$\\to$" in g else 0
    meth = 0 if g.startswith("MS") else (1 if g.startswith("p-term") else 2)
    budget = 0 if "converged" in g else (int(re.search(r"budget (\d+)", g).group(1)) if "budget" in g else 0)
    nodes = int(re.search(r"N(\d+)", g).group(1)) if re.search(r"N(\d+)", g) else 0
    return (kind, meth, 0 if "converged" in g else 1, -budget, nodes, g)
rows = load(); groups = sorted({r["group"] for r in rows}, key=order)
def pretty(g):
    g = g.replace("hp-term", "$hp$-term").replace("p-term", "$p$-term")
    g = g.replace("$hp$-term>MS-term", "$hp$-term$\\,\\to\\,$MS-term").replace("$p$-term>MS-term", "$p$-term$\\,\\to\\,$MS-term")
    g = g.replace(", seed converged", ", converged seed")
    return g
lines = []; L = []
for g in groups:
    rs = [r for r in rows if r["group"] == g]; conv = [r for r in rs if r["converged"]]; ok = [r for r in rs if r["pass"]]
    n = len(rs); it = med([r["iterations"] for r in conv]); seed_it = med([r["seed_it"] for r in conv]); fin_it = med([r["finish_it"] for r in conv])
    t = med([r["scp_time_s"] for r in conv]); cost = [r["cost"] for r in ok if r.get("cost") is not None]
    cost_s = f"{min(cost):.2f}--{max(cost):.2f}" if cost else "--"
    it_s = f"{seed_it:.0f}+{fin_it:.0f}" if rs[0]["kind"] == "hybrid" else f"{it:.0f}"
    print(f"{g:60s} n={n:2d} conv={len(conv):2d} pass={len(ok):2d} it={it_s:8s} t={t:6.1f}s cost(pass)={cost_s}")
    L.append(f"{pretty(g)} & {len(conv)} & {len(ok)} & {it_s} & {t:.1f} & {cost_s} \\\\")
tex = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
       r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.44\textwidth}rrrrr@{}}", r"\toprule",
       r"\textbf{Method} & \textbf{Conv.} & \textbf{Pass} & \textbf{Iter.} & \textbf{Solve (s)} & $\phi_{\mathrm{rep}}(t_F)$ \textbf{(pass)} \\", r"\midrule\addlinespace[2pt]", *L,
       r"\bottomrule", r"\end{tabular}",
       r"\caption{Converged and passing runs out of the 21 initial guesses (the propagated guess and 20 dispersed guesses), with the median iterations (seed stage$+$final stage for hybrid methods) and median solve time over converged runs and the range of replayed crossrange over passing runs. The hybrid methods use a 97-node final stage at $s_y=2$; a seed budget is the iteration count at which the seed stage is stopped and handed off whether or not it has converged.}",
       r"\label{tab:ch6-dispersed}", r"\end{table}"]
os.makedirs(os.path.dirname(OUT_TEX) or ".", exist_ok=True); open(OUT_TEX, "w").write("\n".join(tex) + "\n"); print("wrote", OUT_TEX)

# figure: converged and passing counts per method, and the replayed crossrange of every converged run
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.32 * len(groups) + 1.6), gridspec_kw={"width_ratios": [1.1, 1.0]})
y = np.arange(len(groups))[::-1]
short = [g.replace("-term", "").replace(", seed converged", " (conv.)").replace(", seed budget ", " (b").replace("$s_y=", "$s_y{=}") for g in groups]
short = [x + (")" if " (b" in x and ")" not in x.split(" (b")[-1] else "") for x in short]
nconv = [sum(r["converged"] for r in rows if r["group"] == g) for g in groups]; npass = [sum(r["pass"] for r in rows if r["group"] == g) for g in groups]
axes[0].barh(y + 0.18, nconv, height=0.34, color="0.75", label="converged"); axes[0].barh(y - 0.18, npass, height=0.34, color="C0", label="passed replay")
axes[0].set_yticks(y); axes[0].set_yticklabels(short); axes[0].set_xlabel("runs of %d guesses" % max(sum(1 for r in rows if r["group"] == g) for g in groups)); axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8, frameon=False); axes[0].grid(axis="x", alpha=0.3)
rng = np.random.RandomState(0)
for yi, g in zip(y, groups):
    rs = [r for r in rows if r["group"] == g and r["converged"] and r.get("cost") is not None]
    for r in rs:
        axes[1].plot(r["cost"], yi + rng.uniform(-0.22, 0.22), marker="o", ms=4, mfc=("C0" if r["pass"] else "none"), mec="C0", mew=0.9, ls="none")
axes[1].set_yticks(y); axes[1].set_yticklabels([]); axes[1].set_xlabel(r"$\phi_{\mathrm{rep}}(t_F)$ of converged runs (deg)"); axes[1].grid(axis="x", alpha=0.3)
axes[1].plot([], [], "o", ms=4, color="C0", label="passed"); axes[1].plot([], [], "o", ms=4, mfc="none", mec="C0", label="failed replay"); axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(os.path.splitext(OUT_TEX)[0] + ".pdf"); print("wrote figure")
