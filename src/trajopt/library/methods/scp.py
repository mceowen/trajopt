"""
Subproblem module: SCP with DPP updates 
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import time
import numpy as np

# trajopt imports
from trajopt.library.methods.subproblem import Subproblem
from trajopt.library.methods import hyperparameters as hp


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
    method = trajopt_obj.method

    # Start full problem convergence timer
    time_start = time.perf_counter()

    # Get or create the compiled Subproblem (DPP)
    subprob: Optional[Subproblem] = getattr(method, "subprob", None)
    if subprob is None:
        subprob = Subproblem(problem, method)
        method.subprob = subprob

    # START SUBPROBLEM CONSTRUCTION / HEADER
    print("-" * 152)
    # print(f"                                              ..:: {problem.mission.name}: PTR with Virtual Buffer ::..")
    print("-" * 152)
    print("  Iteration |  Propagation |   Solve   |    Parse   |  log(dz)  |      log(VB)    |   log(VB)   |  log(VB)    | Solve status |  Time of    |   Cost    ")
    print("            |   time [ms]  | time [ms] |  time [ms] |           |  (path + NFZ)   |  (terminal) |  (dynamics) |              |  Flight [s] |           ")
    print("-" * 152)

    max_iter = int(method.conv["iter_max"])

    for _ in range(max_iter + 1):
        subprob.solve_iteration()  # appends a new unified record for this iteration

        latest = subprob.iter_data[-1]
        display_subprob_status(method, latest)

        if latest.get("converged", False):
            print("Terminated from convergence criteria!")
            break

    if not subprob.iter_data[-1].get("converged", False):
        print("Terminated from hitting maximum iterations!")

    trajopt_obj.solution = trajopt_obj.method.subprob.iter_data[-1]

    # Save off convergence time (full time - parse time)
    trajopt_obj.solution['t_full'] = time.perf_counter() - time_start


# ===================================
# Module flags (gating via Parameters)
# ===================================
@dataclass
class ModuleFlags:
    path:  float = 1.0
    nfz:   float = 1.0
    custom:   float = 1.0
    term:  float = 1.0
    dyn:   float = 1.0
    tr:    float = 1.0
    true:  float = 1.0
    dual:  float = 1.0
    vb:    float = 1.0


# ==========================================
# Iteration status printout (unified record)
# ==========================================
def display_subprob_status(method, rec: Dict[str, Any]) -> None:
    conv = rec.get("conv_data", {})

    chk_feas_path = conv.get("chk_feas_path", 0.0)
    chk_feas_nfz  = conv.get("chk_feas_nfz", 0.0)
    ineq_vb       = chk_feas_path + (chk_feas_nfz if chk_feas_nfz != 0 else 0.0)

    chk_dz        = conv.get("chk_dz", 1e-12)
    chk_feas_term = conv.get("chk_feas_term", 1e-12)
    chk_feas_dyn  = conv.get("chk_feas_dyn", 1e-12)

    log_dz      = np.log10(max(chk_dz, 1e-12))
    log_vb_ineq = np.log10(max(ineq_vb, 1e-12))
    log_vb_term = np.log10(max(chk_feas_term, 1e-12))
    log_vb_dyn  = np.log10(max(chk_feas_dyn, 1e-12))

    solve_stat  = conv.get("status", "UNKNOWN")
    iter_num    = int(rec.get("iter_num", -1))

    nt    = float(method.nondim.nt)
    ncost = float(method.nondim.nd_cost)

    Ts   = float(rec.get("T_opt", 0.0))
    cost = float(rec.get("cost", 0.0))

    prop_ms = float(rec.get("prop_time", 0.0) or 0.0)
    solve_ms= float(rec.get("solve_time", 0.0) or 0.0)
    parse_ms= float(rec.get("parse_time", 0.0) or 0.0)

    print(
        "     {:02d}     |    {:07.1f}   |   {:06.1f}  |   {:06.1f}   |   {:+04.1f}    |      {:+05.1f}      |    {:+05.1f}    |     {:+05.1f}   |    {:s}    |   {:4.2f}   |  {:4.1f}".format(
            iter_num,
            prop_ms,
            solve_ms,
            parse_ms,
            log_dz,
            log_vb_ineq,
            log_vb_term,
            log_vb_dyn,
            str(solve_stat),
            Ts * nt,
            cost
        )
    )
