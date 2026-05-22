from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import interp1d

from trajopt.core.problem import Problem
from trajopt.methods.scp import discretize, integrators

if TYPE_CHECKING:
    from trajopt.methods.scp.scvx import SCvx


def set_initial_guess(problem: Problem, method: "SCvx") -> None:
        """Build the initial guess on method, dispatching on the configured type."""
        initial_guess_type = getattr(method.config.method.guess, "type", "propagation")

        if initial_guess_type == "propagation":
            nonlinear_initial_guess(problem, method)

        elif initial_guess_type == "straight_line":
            straight_line_initial_guess(problem, method)

        augment_initial_guess(problem, method)

        method.cost_init = discretize.compute_nonconvex_costs(method.initial_guess.z, method.initial_guess.nu, problem, method)

# TODO(Skye): Possibly add actual computation of beta using index_map helpers
def augment_initial_guess(problem: Problem, method: "SCvx") -> tuple[np.ndarray, np.ndarray]:
    """Assemble the z and nu initial-guess arrays from the t/x/u guess.

    Returns:
        Tuple of (z_init, nu_init).

    """
    idx     = problem.index_map.indices
    N       = method.index_map.N.time_grid

    init                        = method.initial_guess

    t_init                      = np.asarray(init.t).reshape(-1)
    x_init                      = np.asarray(init.x)
    u_init                      = np.asarray(init.u)

    discretize_flag = getattr(method.flags, "discretize", "ms")

    if  discretize_flag == "ps":
        _, etau, _, _ = discretize.compute_ps_differentiation_matrix(N - 1)
        t0, tf = t_init[0], t_init[-1]
        t_lgr  = t0 + (etau + 1.0) / 2.0 * (tf - t0)
        x_init = interp1d(t_init, x_init, axis=0, fill_value="extrapolate")(t_lgr)
        u_init = interp1d(t_init, u_init, axis=0, fill_value="extrapolate")(t_lgr)
        t_init = t_lgr
        init.t = t_init

    z_init                      = np.zeros((N, problem.index_map.n.z))
    z_init[:, idx.z.state]      = x_init
    z_init[:, idx.z.time]       = t_init.reshape(-1, 1)
    z_init[:, idx.z.ctcs]       = 0.0

    nu_init                     = np.zeros((N, problem.index_map.n.nu))
    nu_init[:, idx.nu.control]  = u_init

    dt_ref                      = np.diff(t_init.reshape(-1, 1), axis=0)
    init.dt                     = dt_ref

    if discretize_flag == "ps":
        T                           = float(t_init[-1] - t_init[0])
        nu_init[:, idx.nu.dilation_factor] = T
    else:
        delta_tau                   = 1.0 / (N - 1)
        dt_init                     = np.zeros((N, 1))
        dt_init[:-1, 0]             = dt_ref[:, 0]
        dt_init[-1, 0]              = dt_ref[-1, 0]
        nu_init[:, idx.nu.dilation_factor] = dt_init / delta_tau

    init.z  = z_init
    init.nu = nu_init

    if hasattr(init, "t_dense"):
        t_d  = np.asarray(init.t_dense).reshape(-1)
        x_d  = np.asarray(init.x_dense)
        u_d  = np.asarray(init.u_dense)
        N_d  = len(t_d)
        z_dense              = np.zeros((N_d, problem.index_map.n.z))
        z_dense[:, idx.z.state] = x_d
        z_dense[:, idx.z.time]  = t_d.reshape(-1, 1)
        nu_dense                = np.zeros((N_d, problem.index_map.n.nu))
        nu_dense[:, idx.nu.control] = u_d
        init.z_dense  = z_dense
        init.nu_dense = nu_dense
    else:
        init.z_dense  = z_init
        init.nu_dense = nu_init

    return z_init, nu_init

def straight_line_initial_guess(problem: Problem, method: "SCvx") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a straight-line t/x/u initial guess between boundary states.

    Returns:
        Tuple of (t_init, x_init, u_init).

    """
    init = method.initial_guess

    line_init_u_init            = method.config.method.guess.line_guess_u_init @ problem.nondim.M.control["d2nd"]
    t_init                      = np.asarray(init.t).reshape(-1)


    init_state_constraint       = problem.constraints.get(type="initial_state")[0]
    terminal_state_constraint   = problem.constraints.get(type="final_state")[0]

    if len(init_state_constraint.idx) == problem.index_map.n.state:
        xi_full = init_state_constraint.value
    else:
        xi_guess = getattr(init_state_constraint, "value_guess", None)
        if xi_guess is not None:
            xi_full = xi_guess
        else:
            raise ValueError("Initial_state.xi_guess must be provided for straight_line_initial_guess if initial_state is not fully defined")

    if len(terminal_state_constraint.idx) == problem.index_map.n.state:
        xf_full = terminal_state_constraint.value
    else:
        xf_guess = getattr(terminal_state_constraint, "value_guess", None)
        if xf_guess is not None:
            xf_full = xf_guess
        else:
            raise ValueError("Final_state.xf_guess must be provided for straight_line_initial_guess if final_state is not fully defined")

    # Initial state
    t_start_end_pts     = np.array([t_init[0], t_init[-1]])
    x_start_end_pts     = np.vstack([xi_full, xf_full])
    x_init_interp_func  = interp1d(t_start_end_pts, x_start_end_pts, axis=0)
    x_init              = x_init_interp_func(t_init)

    # Initial control
    u_init_interp_func  = interp1d(t_init, line_init_u_init, axis=0)
    u_init              = u_init_interp_func(t_init)

    init.t       = t_init
    init.x       = x_init
    init.u       = u_init
    init.t_dense = t_init
    init.x_dense = x_init
    init.u_dense = u_init

    return t_init, x_init, u_init

def nonlinear_initial_guess(problem: Problem, method: "SCvx") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a t/x/u initial guess by propagating the nonlinear dynamics.

    Returns:
        Tuple of (t_init, x_nl, u_init).

    """
    init                = method.initial_guess

    x0                  = problem.constraints.get(type="initial_state")[0].value
    dynamics_cnstr      = problem.constraints.get(type="dynamics")[0]

    # ---- Control initialization ----
    nl_init_u_start     = problem.nondim.M.control.d2nd @ method.config.method.guess.nl_guess_u_start
    nl_init_u_stop      = problem.nondim.M.control.d2nd @ method.config.method.guess.nl_guess_u_stop

    t_init              = np.asarray(init.t).reshape(-1)

    # Linearly interpolate control between start and stop values
    t_start_end_pts     = np.array([t_init[0], t_init[-1]])
    u_start_end_pts     = np.vstack([nl_init_u_start, nl_init_u_stop])
    u_init_interp_func  = interp1d(t_start_end_pts, u_start_end_pts, axis=0)
    u_init              = u_init_interp_func(t_init)

    # Propagate physical dynamics dx/dt = f(t, x, u) using fixed-step RK4
    t_nl_out, x_nl, _ = integrators.propagate_txu_rk4(
        x0,
        u_init,
        t_init,
        problem,
        method,
    )

    init.t        = t_init
    init.x        = x_nl
    init.u        = u_init
    init.t_dense  = t_nl_out
    init.x_dense  = x_nl
    init.u_dense  = u_init

    return t_init, x_nl, u_init
