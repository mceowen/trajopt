"""Result tables of the AutoSCvx chapter."""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = {"Quadrotor": HERE, "Unicycle": os.path.join(HERE, "..", "unicycle")}
METHODS = ["Variant 1", "Variant 2a", "Variant 3a", "Variant 2b", "Variant 3b"]
METHODS_TERM = METHODS
V1_NOTES = {"term": r"; Algorithm Variant~1 at the best dual step of its sweep (Table~\ref{tab:ch1-sweep}), $s_{\mathrm{dual}}=5$ on the quadrotor and $10$ on the unicycle",
            "std": r"; Algorithm Variant~1 at the least-bad dual step of its standard-setting screen, $s_{\mathrm{dual}}=0.02$ on the quadrotor and $0.2$ on the unicycle"}
SWEEP_STEPS = [0.5, 1, 2, 5, 10, 20, 50, 100]
GAMMAS = [0.1, 1, 5, 10, 100]
SETTINGS = {"std": dict(compare="compare", dispersed="dispersed", gamma="dispersed_gamma", eq=r"Eq.~\eqref{eq:ch1-setting-std}",
                        name="standard setting", guess="straight-line guess of Eq.~\\eqref{eq:ch1-guess-waypoint}", disp="straight-line guesses through dispersed waypoints"),
            "term": dict(compare="compare_term", dispersed="dispersed_term", gamma="dispersed_term_gamma", eq=r"Eq.~\eqref{eq:ch1-setting-term}",
                         name="terminal-buffered setting", guess="propagated guess of Eq.~\\eqref{eq:ch1-guess-propagated}", disp="guesses propagated from dispersed initial states")}
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "plots", "ch1_tables.tex")

def load(ex, sub, name):
    p = os.path.join(EXAMPLES[ex], "plots", sub, name)
    return json.load(open(p)) if os.path.exists(p) else None

def fmt(x, nd=1):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"

def nominal_table(key):
    S = SETTINGS[key]; L = []
    L += [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{llcccccc}", r"\toprule",
          r"\textbf{Problem} & \textbf{Variant} & \textbf{Conv.} & \textbf{Iter.} & \textbf{SCP (s)} & \textbf{ms/iter.} & \textbf{Objective} & \textbf{Max.\ viol.} \\", r"\midrule"]
    for i, ex in enumerate(EXAMPLES):
        s = load(ex, S["compare"], "summary.json")
        if s is None: continue
        if i: L.append(r"\midrule")
        for j, m in enumerate(METHODS_TERM if key == "term" else METHODS):
            r = s[m]; conv = r["converged"]
            scp = f"{r['scp_time_s']:.2f}" if r.get("scp_time_s") is not None else "--"
            mpi = f"{r['ms_per_iter']:.0f}" if r.get("ms_per_iter") is not None else "--"
            viol = f"{r['max_violation']:.1e}" if r.get("max_violation") is not None else "--"
            L.append(f"{ex if j == 0 else ''} & {m} & {'yes' if conv else 'no'} & {r['iterations']} & {scp} & {mpi} & "
                     f"{fmt(r['objective'] if conv else None)} & {viol} \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{Nominal runs of Algorithm Variants~1, 2a, 3a, 2b, and~3b in the {S['name']} of {S['eq']}, from the nominal {S['guess']}{V1_NOTES[key]}: convergence, SCP iterations, total SCP time and time per iteration, converged objective, and maximum re-propagated violation over the constraints, each in its own units (metres for the dynamics, boundary, and no-fly-zone constraints, newtons for the thrust bound). Objectives of runs converged to different accuracies are not directly comparable.}}",
          rf"\label{{tab:ch1-nominal-{key}}}", r"\end{table}", ""]
    return L

def dispersed_table(key):
    S = SETTINGS[key]; L = []
    L += [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{llccccc}", r"\toprule",
          r"\textbf{Problem} & \textbf{Variant} & \textbf{Converged} & \textbf{Iterations} & \textbf{SCP time (s)} & \textbf{Objective} & \textbf{Max.\ viol.} \\", r"\midrule"]
    for i, ex in enumerate(EXAMPLES):
        t = load(ex, S["dispersed"], "stats.json")
        if t is None: continue
        if i: L.append(r"\midrule")
        for j, m in enumerate(METHODS_TERM if key == "term" else METHODS):
            r = t[m]; n = r["converged"]
            if n:
                L.append(f"{ex if j == 0 else ''} & {m} & {n}/{r['runs']} & ${r['iterations'][0]:.1f} \\pm {r['iterations'][1]:.1f}$ & "
                         f"${r['scp_time_s'][0]:.2f} \\pm {r['scp_time_s'][1]:.2f}$ & ${r['objective'][0]:.1f} \\pm {r['objective'][1]:.1f}$ & {r['max_violation'][0]:.1e} \\\\")
            else:
                L.append(f"{ex if j == 0 else ''} & {m} & {n}/{r['runs']} & -- & -- & -- & -- \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{Statistics over the nominal and 20 dispersed initial guesses in the {S['name']} of {S['eq']} ({S['disp']}){V1_NOTES[key]}: converged runs, and mean $\pm$ standard deviation of the iterations, SCP time and objective, and mean of the maximum re-propagated violation (in the units of Table~\ref{{tab:ch1-nominal-std}}), all over the converged runs.}}",
          rf"\label{{tab:ch1-dispersed-{key}}}", r"\end{table}", ""]
    return L

def sweep_table():
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{l" + "c" * len(SWEEP_STEPS) + "}", r"\toprule",
         r"\textbf{Problem} & " + " & ".join(f"$s_{{\\mathrm{{dual}}}}={g:g}$" for g in SWEEP_STEPS) + r" \\", r"\midrule"]
    for ex in EXAMPLES:
        t = load(ex, "sweep_term", "summary.json")
        cells = []
        for g in SWEEP_STEPS:
            r = None if t is None else t.get(f"{g:g}")
            cells.append("--" if r is None else (f"{r['iterations']}" if r["converged"] else "no"))
        L.append(f"{ex} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Algorithm Variant~1 in the terminal-buffered setting of Eq.~\eqref{eq:ch1-setting-term}, from the nominal propagated guess: iterations to convergence for each dual step $s_{\mathrm{dual}}$, within a budget of 2000 iterations (no: not converged within the budget).}",
          r"\label{tab:ch1-sweep}", r"\end{table}"]
    return L

TUNED = [("Quadrotor", "std", "dispersed_tuned", "Variant 3a", r"$s_z=0.2$"),
         ("Unicycle", "std", "dispersed_tuned", "Variant 2b", r"$s_z=1$, $\kappa_{\varepsilon}=0.1$")]

def tuned_table():
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lllccccc}", r"\toprule",
         r"\textbf{Problem} & \textbf{Setting} & \textbf{Variant} & \textbf{Changed setting} & \textbf{Common} & \textbf{Tuned} & \textbf{Iterations} & \textbf{Max.\ viol.} \\", r"\midrule"]
    for ex, key, sub, v, changed in TUNED:
        c = load(ex, SETTINGS[key]["dispersed"], "stats.json")[v]; t = load(ex, sub, "stats.json")[v]
        L.append(f"{ex} & {'standard' if key == 'std' else 'terminal-buffered'} & {v} & {changed} & {c['converged']}/{c['runs']} & {t['converged']}/{t['runs']} & "
                 f"${t['iterations'][0]:.1f} \\pm {t['iterations'][1]:.1f}$ & {t['max_violation'][0]:.1e} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{Variants that fail at the common settings, rerun over the same nominal and 20 dispersed guesses with the one setting that lets them converge: converged runs at the common and at the tuned setting, and the mean $\pm$ standard deviation of the iterations and the mean maximum re-propagated violation of the tuned runs. All other settings are those of Table~\ref{tab:ch1-setup}.}",
          r"\label{tab:ch1-tuned}", r"\end{table}"]
    return L

CELLS = [("standard (dynamics buffered)", "straight-line", "dispersed"), ("standard (dynamics buffered)", "propagated", "dispersed_std_prop"),
         ("terminal-buffered (dynamics exact)", "straight-line", "dispersed_term_wp"), ("terminal-buffered (dynamics exact)", "propagated", "dispersed_term")]

def cells_table():
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lll" + "c" * len(METHODS) + "}", r"\toprule",
         r"\textbf{Problem} & \textbf{Buffering} & \textbf{Guess} & " + " & ".join(r"\textbf{" + m.replace("Variant ", "") + "}" for m in METHODS) + r" \\", r"\midrule"]
    for i, ex in enumerate(EXAMPLES):
        if i: L.append(r"\midrule")
        for j, (buff, guess, sub) in enumerate(CELLS):
            t = load(ex, sub, "stats.json"); cells = []
            for m in METHODS:
                r = None if t is None else t.get(m)
                cells.append("--" if r is None else f"{r['converged']}/{r['runs']}" + (f" ({r['iterations'][0]:.0f})" if r["converged"] else ""))
            L.append(f"{ex if j == 0 else ''} & {buff.split(' ')[0] if j % 2 == 0 else ''} & {guess} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{The full design: both bufferings crossed with both guesses, over the nominal and 20 dispersed guesses of each kind (Table~\ref{tab:ch1-setup}), for Algorithm Variants~1, 2a, 3a, 2b, and~3b: converged runs out of 21 and, in parentheses, the mean number of iterations of the converged runs. The first and last rows of each problem are the two settings of Section~\ref{subsec:ch1-settings}, repeated from Tables~\ref{tab:ch1-dispersed-std} and~\ref{tab:ch1-dispersed-term}.}",
          r"\label{tab:ch1-cells}", r"\end{table}"]
    return L

lines = sweep_table() + nominal_table("std") + dispersed_table("std") + nominal_table("term") + dispersed_table("term") + tuned_table() + cells_table()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(lines) + "\n")
print("wrote", OUT)
