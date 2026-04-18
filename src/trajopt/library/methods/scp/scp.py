"""
Subproblem module: SCP with DPP updates 
"""

from __future__ import annotations
from typing import Dict, Any
import time
import numpy as np

# trajopt imports


# =====================================================================================
# Public API: full Sequential Convex Programming (SCP) loop with DPP reuse
# =====================================================================================
def run_scp(trajopt_obj):
    """
    Run the full Sequential Convex Programming (SCP) loop for a given trajopt_obj.

    Subproblem maintains iteration history internally:
      - subprob.iter_data[0]: initialization inputs only
      - subprob.iter_data[it>=1]: inputs used at iter it + outputs from that iter
    """

    problem = trajopt_obj.problem

    # Start full problem convergence timer
    time_start = time.perf_counter()

    # subproblem convergence header
    print("-" * 164)
    print("  Iteration |  Propagation |   Solve   |    Parse   |  log(dx/eps) | log(vb_ineq/eps) | log(vb_term/eps) | log(vb_dyn/eps) | Solve status |  Time of    |   Cost    ")
    print("            |   time [ms]  | time [ms] |  time [ms] |     (state)  |    (ncvx_ineq)   |      (terminal)  |    (dynamics)   |              |  Flight [s] |           ")
    print("-" * 164)

    max_iter = int(trajopt_obj.method.conv.iter_max)

    for _ in range(max_iter + 1):
        trajopt_obj.method.subproblem.solve_iteration()  # appends a new unified record for this iteration

        latest = trajopt_obj.method.subproblem.iter_data[-1]
        display_subprob_status(trajopt_obj.method, latest)

        if latest.get("converged", False):
            print("Terminated from convergence criteria!")
            break

    if not trajopt_obj.method.subproblem.iter_data[-1].get("converged", False):
        print("Terminated from hitting maximum iterations!")

    trajopt_obj.solution = trajopt_obj.method.subproblem.iter_data[-1]

    # Save off convergence time (full time - parse time)
    trajopt_obj.solution['t_full'] = time.perf_counter() - time_start


# ==========================================
# Iteration status printout (unified record)
# ==========================================
def display_subprob_status(method, rec: Dict[str, Any]) -> None:
    conv = rec.get("conv_data", {})

    with np.errstate(divide='ignore'):
        log_dz_ratio      = np.log10(conv["chk_dz"])
        log_vb_ineq_ratio = np.log10(conv["chk_vb_ineq"])
        log_vb_term_ratio = np.log10(conv["chk_vb_term"])
        log_vb_dyn_ratio  = np.log10(conv["chk_vb_dyn"])

    solve_stat  = conv.get("status", "UNKNOWN")
    iter_num    = int(rec.get("iter_num", -1))

    Ts   = float(rec.get("T_opt", 0.0))
    cost = float(rec.get("cost", 0.0))

    prop_ms = float(rec.get("prop_time", 0.0) or 0.0)
    solve_ms= float(rec.get("solve_time", 0.0) or 0.0)
    parse_ms= float(rec.get("parse_time", 0.0) or 0.0)

    print(
        "{:^12d}|{:^14.1f}|{:^11.1f}|{:^12.1f}|{:^+14.1f}|{:^+18.1f}|{:^+18.1f}|{:^+17.1f}|{:^14s}|{:^13.2f}|{:^11.1f}".format(
            iter_num,
            prop_ms,
            solve_ms,
            parse_ms,
            log_dz_ratio,
            log_vb_ineq_ratio,
            log_vb_term_ratio,
            log_vb_dyn_ratio,
            str(solve_stat),
            Ts * method.nondim.time_scale,
            cost
        )
    )