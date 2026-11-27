"""Mesh refinement of multiple shooting and the hp-method at the selected proximal steps: accuracy of the replay against mesh size."""
import glob, json, os, sys
import numpy as np
OUT_TEX = sys.argv[1] if len(sys.argv) > 1 else "plots/refine_table.tex"
rows = []
for f in sorted(glob.glob("plots/matrix_MS-term/screen.json") + glob.glob("plots/matrix_ms_fine/screen.json")):
    rows += [r for r in json.load(open(f)) if r.get("checks") and r["sz"] == 4.0 and r["N"] in (49, 97, 129, 193, 257, 385)]
for f in ("plots/matrix_hp-term/screen.json", "plots/matrix_hp97/screen.json", "plots/matrix_hp_fine/screen.json", "plots/matrix_hp_fine2/screen.json"):
    rows += [r for r in json.load(open(f)) if r.get("checks") and r["control"] == "foh" and r["stride"] == 2 and r["sz"] == 2.0 and (r["N"] - 1) // r["K"] == 6 and r["K"] in (8, 16, 24, 32, 48, 64)]
def sci(x):
    if x == 0: return "0"
    e = int(np.floor(np.log10(abs(x)))); m = x / 10**e
    return f"${m:.1f}\\times10^{{{e}}}$"
L = []
for m, label in (("MS-term", "MS-term"), ("hp-term", "$hp$-term")):
    for r in sorted([r for r in rows if r["method"] == m], key=lambda r: r["N"]):
        c = r["checks"]; path = max(c[k]["v"] / c[k]["a"] for k in ("heat rate", "dynamic pressure", "load"))
        mesh = f"{r['N']}" if m == "MS-term" else f"{r['N']} ({r['K']}$\\times$6)"
        L.append(f"{label} & {mesh} & {r['iterations']} & {r['scp_time_s']:.1f} & ${r['cost']:.2f}^\\circ$ & {sci(c['dynamics']['v'])} & {sci(c['endpoint drift']['v'])} & {path:.2f} & {r['R']:.2f} & {'Pass' if r['pass'] else 'Fail'} \\\\")
        print(L[-1])
    L.append(r"\addlinespace[2pt]")
tex = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
       r"\resizebox{\textwidth}{!}{\begin{tabular}{@{}llrrrrrrrl@{}}", r"\toprule",
       r"\textbf{Method} & \textbf{Nodes} & \textbf{Iter.} & \textbf{Solve (s)} & $\phi_{\mathrm{rep}}(t_F)$ & \textbf{Dynamics} & \textbf{Drift} & \textbf{Path} & $R$ & \textbf{Replay} \\", r"\midrule\addlinespace[2pt]", *L[:-1],
       r"\bottomrule", r"\end{tabular}}",
       r"\caption{Mesh refinement of multiple shooting ($s_y=4$) and of the hp-method with intervals of degree six ($s_y=2$, FOH 2) from the propagated initial guess. \emph{Dynamics} is the largest scaled discrepancy between a node and the state propagated from the preceding node and \emph{Drift} the scaled distance between the replayed endpoint and the last node, both with allowance $0.001$; \emph{Path} is the largest path-constraint exceedance between nodes as a fraction of its $2\%$ allowance.}",
       r"\label{tab:ch6-refinement}", r"\end{table}"]
open(OUT_TEX, "w").write("\n".join(tex) + "\n"); print("wrote", OUT_TEX)
