"""Bare powered-descent problem (no altitude-triggered constraints): standalone multiple shooting (15 nodes, dynamics buffered),
p-method (9 nodes, terminal-buffered) and hp-method (41 nodes, 10 intervals) from the straight-line guess and dispersed guesses,
with an acceptance test on the evaluation diagnostics: propagated-endpoint error and constraint residuals."""
import argparse, json, os
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from trajopt.trajectory_analyzer import TrajectoryAnalyzer
from trajopt.analysis.experiments import ExperimentCase, ExperimentVariant, run_experiment, monte_carlo_cases
from trajopt.analysis.experiments import _build_solver

MESH = {"MS": dict(num_nodes=15), "p": dict(num_nodes=9), "hp": dict(num_nodes=41, hp_segments=10), "hp-term": dict(num_nodes=41, hp_segments=10)}
ALLOW = {"min_thrust": 0.02 * 0.3, "final_state": 2e-3, "initial_state": 1e-3, "final_time": 1e-3}   # 2% of the thrust bound; 2% of the unit scale for the dimensionless convex residuals
DEFAULT_ALLOW = 0.02
ENDPOINT_ALLOW = 2e-3   # the final stage buffers its dynamics with tolerance 1e-4 per interval over 14 intervals
VARIATIONS = {"seed": 0, "num": 20, "samples": {
    "method.hyperparams.initial_guess.default.x_start": {"type": "uniform", "lb": [0, -1.0, -1.0, -1.0, -0.5, -0.5, -0.5, 0, 0, 0, 0, 0, 0, 0], "ub": [0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0]},
    "method.hyperparams.initial_guess.default.u_start": {"type": "uniform", "lb": [-0.5, -0.5, -0.5], "ub": [0.5, 0.5, 0.5]},
    "method.hyperparams.initial_guess.default.u_stop": {"type": "uniform", "lb": [-0.5, -0.5, -0.5], "ub": [0.5, 0.5, 0.5]}}}

def with_trajectory(**updates):
    def factory(config):
        for k, v in updates.items(): config.trajectory[k] = v
        return _build_solver(config)
    return factory

def accept(run):
    """Acceptance from the evaluation diagnostics of the (last) phase."""
    ev = run.evaluation
    if ev is None or not ev.phases: return dict(R=None, limiting="no evaluation", cost=None, t_final=None, checks={})
    ph = next(iter(ev.phases.values())); it, dg = ph.final, ph.diagnostics
    checks = {"endpoint": (float(np.max(np.abs(np.asarray(dg["endpoint_state_error"])))), ENDPOINT_ALLOW)}
    for name, res in dict(dg["physical_constraint_violations"]).items():
        checks[name] = (float(res["maximum"]), ALLOW.get(name, DEFAULT_ALLOW))
    ratios = {k: v / a for k, (v, a) in checks.items()}; lim = max(ratios, key=ratios.get)
    return dict(R=float(ratios[lim]), limiting=lim, cost=float(np.asarray(it.t_opt)[-1]), t_final=float(np.asarray(it.t_opt)[-1]),
                checks={k: dict(v=v, a=a) for k, (v, a) in checks.items()})

def row_of(run, extra):
    r = run.solve_result
    row = dict(extra, case=run.case, converged=bool(getattr(r, "converged", False)), iterations=int(getattr(r, "iterations", 0) or 0), status=str(getattr(r, "status", None)),
               scp_time_s=float(dict(getattr(r, "timings", {}) or {}).get("total_s", float("nan"))), wall_s=float(run.wall_s), error=run.error)
    try: row.update(accept(run))
    except Exception as e: row.update(R=None, limiting=f"evaluation error: {e}", cost=None)
    row["pass"] = bool(row["converged"] and row.get("R") is not None and row["R"] <= 1.0)
    return row

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--methods", nargs="*", default=["MS", "p", "hp", "hp-term"]); ap.add_argument("--config", default="pdg_compare.yaml")
    ap.add_argument("--out", default="plots/pdg_matrix"); ap.add_argument("--dispersed", type=int, default=0)
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    traj = TrajectoryAnalyzer(args.config); cfg = traj.config
    if args.dispersed:
        cfg["variations"] = dict(VARIATIONS, num=args.dispersed); cases = list(monte_carlo_cases(cfg))
        def _get(d, path):
            for k in path.split("."): d = d[k]
            return d
        json.dump({c.name: {k: _get(c.config, "method.hyperparams.initial_guess.default." + k) for k in ("x_start", "u_start", "u_stop")} for c in cases},
                  open(os.path.join(args.out, "cases.json"), "w"), indent=1, default=lambda v: np.asarray(v).tolist())
    else:
        cases = [ExperimentCase("nominal", cfg, nominal=True)]
    rows = []
    for m in args.methods:
        v = ExperimentVariant(m, cfg.analysis.methods[m] or {}, operation_factory=with_trajectory(**MESH[m]), operation_id=f"SCPMethod:{m}")
        exp = run_experiment(cases, [v], evaluate=True, analysis_config=cfg.analysis)
        for run in exp.runs:
            row = row_of(run, dict(method=m, tag=m + ("" if run.case == "nominal" else f"|{run.case}"), **MESH[m]))
            rows.append(row); print(f"RESULT {row['tag']} conv={row['converged']} it={row['iterations']} scp={row['scp_time_s']:.1f}s tF={row.get('cost')} R={row.get('R')} limiting={row.get('limiting')} pass={row['pass']}", flush=True)
            json.dump(rows, open(os.path.join(args.out, "screen.json"), "w"), indent=1, default=str)
