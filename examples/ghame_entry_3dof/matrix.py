"""Tuning screen over mesh and proximal steps for the standalone MS, p, and hp transcriptions, with replay checks."""
import argparse, copy, json, os, time
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from trajopt.trajectory_analyzer import TrajectoryAnalyzer
from trajopt.analysis.experiments import ExperimentCase, ExperimentVariant, run_experiment, monte_carlo_cases
from variants import with_trajectory
from replay import replay, PATH_ALLOW

PS = "trajopt/methods/common/configs/discretize/pseudospectral_p.yaml"
STEPS = [(1.0, 0.05), (2.0, 0.1), (4.0, 0.2)]
MS_NODES = [33, 49, 65, 97]
P_NODES = [25, 33, 41, 49]
HP_MESH = [(4, 6), (6, 6), (7, 7), (8, 6), (12, 4)]
CONTROLS = [("lagrange", 1), ("foh", 1), ("foh", 2)]
HP_KNOTS = None


def configs(method, steps=None):
    out = []
    for (sz, su) in (steps or STEPS):
        if method == "MS-term":
            for N in MS_NODES:
                out.append(dict(method=method, N=N, K=1, control="foh", stride=1, sz=sz, su=su))
        elif method == "p-term":
            for N in P_NODES:
                for ci, st in CONTROLS:
                    if ci == "foh" and (N - 1) % st: continue
                    out.append(dict(method=method, N=N, K=1, control=ci, stride=st, sz=sz, su=su))
        else:
            for K, d in HP_MESH:
                for ci, st in CONTROLS:
                    if ci == "foh" and d % st: continue
                    out.append(dict(method=method, N=K * d + 1, K=K, control=ci, stride=st, sz=sz, su=su))
    return out


def tag(c):
    return f"{c['method']}|N{c['N']}|K{c['K']}|{c['control']}{c['stride']}|sz{c['sz']:g}" + ("|knots" if HP_KNOTS else "")


def variant(c, ps_base):
    from trajopt.common.tools import deep_merge, recursive_attrdict
    steps = {"hyperparams": {"trust_region": {"tr_step": {"z": {"default": c["sz"]}, "nu": {"default": c["su"], "u": c["su"], "s": 1.0}}}}}
    if c["method"] == "MS-term":
        ov = recursive_attrdict(steps)
        factory = with_trajectory(num_nodes=c["N"])
    else:
        disc = {"hyperparams": {"discretize": {"control_interpolation": c["control"], "control_stride": c["stride"]}}}
        ov = deep_merge(deep_merge(recursive_attrdict(copy.deepcopy(dict(ps_base))), recursive_attrdict(disc)), recursive_attrdict(steps))
        factory = with_trajectory(num_nodes=c["N"], hp_segments=c["K"], **({"hp_knots": list(HP_KNOTS)} if HP_KNOTS else {}))
    return ExperimentVariant(tag(c), ov, operation_factory=factory, operation_id=f"SCPMethod:{tag(c)}")


def limits_from(cfg, built):
    fcns, params = built.fcns, built.params
    tr = cfg.trajectory
    R = float(params.planet.r)
    return dict(dynamics=fcns.dynamics, params=params, fcns=fcns,
                path_fcns={"heat rate": fcns.heat_rate, "dynamic pressure": fcns.dynamic_pressure, "load": fcns.aero_load},
                path={"heat rate": float(tr.constraints.max_heat_rate.upper[0]), "dynamic pressure": float(tr.constraints.max_dynamic_pressure.upper[0]), "load": float(tr.constraints.max_aero_load.upper[0])},
                terminal={"altitude": (0, R + 15000.0, 1e5), "flight-path angle": (4, -10.0, 10.0)},
                state_bounds=list(zip([float(v) for v in tr.constraints.state_limits.lower], [float(v) for v in tr.constraints.state_limits.upper])),
                state_names=["r", "theta", "phi", "v", "gamma", "psi"],
                control_bounds=list(zip([float(v) for v in tr.constraints.control_limits.lower], [float(v) for v in tr.constraints.control_limits.upper])),
                control_names=["bank", "aoa"], control_rate=[float(v) for v in tr.constraints.control_rate_limit.value])


SCALES = np.array([1e5, 90.0, 90.0, 1e4, 10.0, 90.0])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--methods", nargs="*", default=["MS-term", "p-term", "hp-term"])
    ap.add_argument("--config", default="compare_all.yaml"); ap.add_argument("--out", default="plots/matrix")
    ap.add_argument("--only", default=None, help="run a single tag"); ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--save-iters", action="store_true")
    ap.add_argument("--limits-config", default=None, help="config whose path limits the replay checks against")
    ap.add_argument("--ms-nodes", nargs="*", type=int, default=None)
    ap.add_argument("--p-nodes", nargs="*", type=int, default=None); ap.add_argument("--hp-mesh", nargs="*", type=int, default=None, help="alternating K N_k pairs")
    ap.add_argument("--controls", nargs="*", default=None, help="control interpolations to screen, e.g. foh2")
    ap.add_argument("--hp-knots", nargs="*", type=float, default=None, help="knot fractions of the hp intervals (len K+1); equal intervals when omitted")
    ap.add_argument("--dispersed", type=int, default=0, help="number of dispersed initial guesses in addition to the nominal one (seed 0; bank angle and angle of attack of the guess within +-20 and +-3 deg at its start and end, duration within +-200 s)")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--steps", nargs="*", type=float, default=None, help="alternating s_y s_nu pairs")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    if args.ms_nodes: MS_NODES[:] = args.ms_nodes
    if args.p_nodes: P_NODES[:] = args.p_nodes
    if args.hp_mesh: HP_MESH[:] = list(zip(args.hp_mesh[::2], args.hp_mesh[1::2]))
    if args.hp_knots: HP_KNOTS = list(args.hp_knots)
    if args.controls: CONTROLS[:] = [(c.rstrip("0123456789"), int(c[len(c.rstrip("0123456789")):] or 1)) for c in args.controls]
    traj = TrajectoryAnalyzer(args.config); cfg = traj.config
    lt = TrajectoryAnalyzer(args.limits_config) if args.limits_config else traj
    limits = limits_from(lt.config, list(lt.problem.phases.values())[0].build())
    ps_base = cfg.analysis.methods["p-method"]
    if args.dispersed:
        cfg["variations"] = {"seed": args.rng_seed, "num": args.dispersed, "samples": {
            "method.hyperparams.initial_guess.default.u_start": {"type": "uniform", "lb": [-20.0, -3.0], "ub": [20.0, 3.0]},
            "method.hyperparams.initial_guess.default.u_stop": {"type": "uniform", "lb": [-20.0, -3.0], "ub": [20.0, 3.0]},
            "method.hyperparams.initial_guess.default.t_stop": {"type": "uniform", "lb": -200.0, "ub": 200.0}}}
        cases = list(monte_carlo_cases(cfg))
        def _get(d, path):
            for k in path.split("."): d = d[k]
            return d
        os.makedirs(args.out, exist_ok=True)
        json.dump({cs_.name: {k: _get(cs_.config, "method.hyperparams.initial_guess.default." + k) for k in ("u_start", "u_stop", "t_stop")} for cs_ in cases},
                  open(os.path.join(args.out, "cases.json"), "w"), indent=1, default=lambda v: np.asarray(v).tolist())
    else:
        cases = [ExperimentCase("nominal", cfg, nominal=True)]
    rows = []
    for m in args.methods:
        steps = list(zip(args.steps[0::2], args.steps[1::2])) if args.steps else None
        cs = [c for c in configs(m, steps) if args.only is None or tag(c) == args.only]
        if not cs: continue
        exp = run_experiment(cases, [variant(c, ps_base) for c in cs], evaluate=True, analysis_config=cfg.analysis)
        by_tag = {tag(c): c for c in cs}
        for run in exp.runs:
            c = by_tag[run.variant]; r = run.solve_result
            rtag = tag(c) + ("" if run.case == "nominal" else f"|{run.case}")
            row = dict(c, tag=rtag, case=run.case, converged=bool(getattr(r, "converged", False)), iterations=int(getattr(r, "iterations", 0) or 0),
                       attempts=(len(r.attempts) if hasattr(getattr(r, "attempts", None), "__len__") else int(getattr(r, "attempts", 0) or 0)), status=str(getattr(r, "status", None)),
                       scp_time_s=float(dict(getattr(r, "timings", {}) or {}).get("total_s", float("nan"))), wall_s=float(run.wall_s), error=run.error)
            if run.evaluation is not None and run.evaluation.phases:
                try:
                    rep = replay(run, limits, SCALES, "ms" if m == "MS-term" else "ps", c["K"], c["control"], c["stride"], samples=args.samples, knots=HP_KNOTS if m != "MS-term" else None)
                    row.update(R=rep["R"], limiting=rep["limiting"], cost=rep["cost"], nodal_cost=rep["nodal_cost"], t_final=rep["t_final"], checks=rep["checks"], path_allowance=PATH_ALLOW)
                    np.savez(os.path.join(args.out, rtag.replace("|", "_") + ".npz"), **{f"dense_{k}": v for k, v in rep["dense"].items() if k != "path"},
                             **{f"path_{k.replace(' ', '_')}": v for k, v in rep["dense"]["path"].items()}, **{f"nodes_{k}": v for k, v in rep["nodes"].items()})
                except Exception as e:
                    row.update(R=None, limiting=f"replay error: {e}", cost=None)
            row["pass"] = bool(row["converged"] and row.get("R") is not None and row["R"] <= 1.0)
            if args.save_iters and getattr(r, "phase_data", None):
                its = [d for d in next(iter(r.phase_data.values())).iter_data_list if d.get("chk") is not None]
                hist = {k: np.array([float(d["chk"].get(k, 0.0)) for d in its]) for k in ("dz", "final_state", "nonconvex_inequality", "dynamics")}
                hist["cost"] = np.array([float(d.get("cost", np.nan)) for d in its]); hist["penalty_cost"] = np.array([float(d.get("penalty_cost", 0.0) or 0.0) for d in its])
                hist["objective"] = np.array([float(d.get("x_opt")[-1][2]) if d.get("x_opt") is not None else np.nan for d in its])
                np.savez(os.path.join(args.out, rtag.replace("|", "_") + "_iters.npz"), **hist)
            rows.append(row)
            print(f"RESULT {row['tag']} conv={row['converged']} it={row['iterations']} scp={row['scp_time_s']:.1f}s cost={row.get('cost')} R={row.get('R')} limiting={row.get('limiting')} pass={row['pass']}")
            json.dump(rows, open(os.path.join(args.out, "screen.json"), "w"), indent=1, default=str)
