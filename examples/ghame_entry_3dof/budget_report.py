"""Seed-budget sweep on the nominal guess: hybrid methods whose seed stage is stopped after a fixed number of iterations,
against the converged-seed hybrids and standalone multiple shooting at the same final-stage settings."""
import glob, json, re, sys
import numpy as np
OUT_TEX = sys.argv[1] if len(sys.argv) > 1 else "plots/budget_table.tex"
rows = []
for f in glob.glob("plots/budget/*/row.json") + glob.glob("plots/handoff/*_to_MS-term_N97_sz[24]/row.json"):
    r = json.load(open(f)); b = re.search(r"\|b(\d+)", r["tag"]); st = r["stages"]
    rows.append(dict(seed=r["method"].split(">")[0].replace("-term", ""), budget=int(b.group(1)) if b else None, sz=r["sz"], seed_it=st["seed"]["iterations"], seed_conv=st["seed"]["converged"],
                     fin_it=st.get("finish", {}).get("iterations"), fin_conv=st.get("finish", {}).get("converged", False), t=r["scp_time_s"], seed_t=st["seed"]["solve_s"], cost=r.get("cost"), R=r.get("R"), ok=r["pass"]))
ms = [r for r in json.load(open("plots/matrix_MS-term/screen.json")) if r["N"] == 97 and r["sz"] in (2.0, 4.0) and r.get("checks")]
L = []
for r in sorted(ms, key=lambda r: r["sz"]):
    L.append(f"MS-term & -- & {r['sz']:g} & -- & {r['iterations']} & {r['scp_time_s']:.1f} & ${r['cost']:.3f}^\\circ$ & {r['R']:.2f} & {'Pass' if r['pass'] else 'Fail'} \\\\"); print(L[-1])
L.append(r"\addlinespace[2pt]")
for seed in ("p", "hp"):
    for r in sorted([r for r in rows if r["seed"] == seed], key=lambda r: (r["budget"] or 10**6, r["sz"])):
        bud = "converged" if r["budget"] is None else str(r["budget"])
        fin = f"{r['fin_it']}" if r["fin_it"] is not None else "--"
        L.append(f"${seed}$-term$\\,\\to\\,$MS-term & {bud} & {r['sz']:g} & {r['seed_it']} & {fin} & {r['t']:.1f} & " + (f"${r['cost']:.3f}^\\circ$ & {r['R']:.2f}" if r["cost"] is not None and r["fin_conv"] else "-- & --") + f" & {'Pass' if r['ok'] else 'Fail'} \\\\"); print(L[-1])
    L.append(r"\addlinespace[2pt]")
tex = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
       r"\resizebox{\textwidth}{!}{\begin{tabular}{@{}llrrrrrrl@{}}", r"\toprule",
       r"\textbf{Method} & \textbf{Seed budget} & $s_y$ & \textbf{Seed iter.} & \textbf{Final iter.} & \textbf{Solve (s)} & $\phi_{\mathrm{rep}}(t_F)$ & $R$ & \textbf{Replay} \\", r"\midrule\addlinespace[2pt]", *L[:-1],
       r"\bottomrule", r"\end{tabular}}",
       r"\caption{Seed-budget sweep from the propagated initial guess: the seed stage (49 nodes, the selected collocation settings) is stopped after the listed number of iterations and handed off to a 97-node multiple-shooting final stage whether or not it has converged; ``converged'' rows let the seed stage run to convergence. Solve time is the total of both stages in one run. Standalone multiple shooting at the same final-stage settings is listed for reference.}",
       r"\label{tab:ch6-budget}", r"\end{table}"]
open(OUT_TEX, "w").write("\n".join(tex) + "\n"); print("wrote", OUT_TEX)
