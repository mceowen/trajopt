import trajopt.methods.scp as scp
from trajopt.methods.scp.subproblem import Subproblem

from typing import Dict, Any
import numpy as np

def initialize(method):
    problem = method.problem
    
    # nondimensionalize and convexfiy constraints
    problem.constraints.nondim_constraints(method.nondim)
    problem.constraints.convexify_constraints()

    # nondimensionalize and convexify costs
    problem.costs.nondim_costs(method.nondim)
    problem.costs.convexify_costs()

    # ---- Time grid initialization ----
    method.Ts_init  = method.initial_guess.T_init / method.nondim.time_scale
    t_init = np.linspace(0.0, method.Ts_init, method.index_map.N.time_grid).reshape(-1, 1)
    dt_init = np.diff(t_init, axis=0)
    method.initial_guess.t = t_init.reshape(-1)
    method.initial_guess.dt = dt_init

    scp.discretize.compile_jax_discretization(problem, method)

    ### Time of flight constraints ###
    method.Ts_min  = method.initial_guess.T_min / method.nondim.time_scale
    method.Ts_max  = method.initial_guess.T_max / method.nondim.time_scale
    method.ddt_max = method.initial_guess.dT_max / ((method.index_map.N.time_grid - 1) * method.nondim.time_scale)
    method.dt_min  = method.Ts_min / (method.index_map.N.time_grid - 1)
    method.dt_max  = method.Ts_max / (method.index_map.N.time_grid - 1)

    scp.hyperparameters.configure_penalty_weights(problem, method)

    ### Configure generic convergence criterion and max iterations ###
    scp.convergence.set_convergence_tolerance(problem, method)

    scp.initial_guess.set_initial_guess(problem, method)

    # TODO(Skye/Carlos): Potentially move the method=specific constraint modeling here 
    # (instead of the subproblem)
    # OR maybe just initialize the subproblem itself here and only run solve later?
    # --- Initialize virtual buffers ---
    method.conv_data.vb_ineq = np.zeros((method.index_map.N.time_grid, method.index_map.n.nonconvex_inequality))
    method.conv_data.vb_dyn = np.zeros((method.index_map.N.time_grid-1, method.index_map.n.z))
    method.conv_data.vb_terminal = np.zeros(method.index_map.n.z)

    # --- Initialize reusable compiled Subproblem (DPP) ---
    method.subproblem = Subproblem(problem, method)

# =====================================================================================
# Public API: full Sequential Convex Programming (SCP) loop with DPP reuse
# =====================================================================================
def run_scp(method):
    """
    Run the full Sequential Convex Programming (SCP) loop for a given trajopt_obj.

    Subproblem maintains iteration history internally:
      - subprob.iter_data[0]: initialization inputs only
      - subprob.iter_data[it>=1]: inputs used at iter it + outputs from that iter
    """

    # subproblem convergence header
    print("-" * 164)
    print("  Iteration |  Propagation |   Solve   |    Parse   |  log(dx/eps) | log(vb_ineq/eps) | log(vb_term/eps) | log(vb_dyn/eps) | Solve status |  Time of    |   Cost    ")
    print("            |   time [ms]  | time [ms] |  time [ms] |     (state)  |    (ncvx_ineq)   |      (terminal)  |    (dynamics)   |              |  Flight [s] |           ")
    print("-" * 164)

    max_iter = int(method.conv.iter_max)

    for _ in range(max_iter + 1):
        method.subproblem.solve_iteration()  # appends a new unified record for this iteration

        latest_iter_data = method.subproblem.iter_data[-1]
        display_subprob_status(latest_iter_data)

        if latest_iter_data.converged:
            print("Terminated from convergence criteria!")
            break

    if not method.subproblem.iter_data[-1].converged:
        print("Terminated from hitting maximum iterations!")

    solution = method.subproblem.iter_data[-1]

    return solution


# ==========================================
# Iteration status printout (unified record)
# ==========================================
def display_subprob_status(rec: Dict[str, Any]) -> None:
    conv = rec["conv_data"]

    with np.errstate(divide='ignore'):
        log_dz_ratio      = np.log10(conv["chk_dz"])
        log_vb_ineq_ratio = np.log10(conv["chk_vb_ineq"])
        log_vb_term_ratio = np.log10(conv["chk_vb_term"])
        log_vb_dyn_ratio  = np.log10(conv["chk_vb_dyn"])

    solve_stat  = conv.status
    iter_num    = rec.iter_num

    T    = rec.T_opt
    cost = rec.cost

    prop_ms = rec.prop_time
    solve_ms= rec.solve_time
    parse_ms= rec.parse_time

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
            T,
            cost
        )
    )