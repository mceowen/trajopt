import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp

from trajopt.methods.scp import discretize


CREATE_PARAMETER_REGISTRY:         dict = {}
CREATE_VARIABLE_REGISTRY:          dict = {}
CREATE_CONSTRAINT_REGISTRY:        dict = {}
CREATE_COST_REGISTRY:              dict = {}
UPDATE_PARAMETER_REGISTRY:         dict = {}
UPDATE_CURRENT_ITER_DATA_REGISTRY: dict = {}
INITIALIZE_METHOD_REGISTRY:        dict = {}


# =============================================================================
# DYNAMICS
# =============================================================================

def create_cvxpy_parameters_dynamics(method):

    N, n_z, n_nu = method.N.time_grid, method.n.z, method.n.nu

    if method.flags.discretize == "ms":
        method.cp_params.Ak  = cp.Parameter((N - 1, n_z, n_z), name="Ak")
        method.cp_params.Bk  = cp.Parameter((N - 1, n_z, n_nu), name="Bk")
        method.cp_params.Bkp = cp.Parameter((N - 1, n_z, n_nu), name="Bkp")
        method.cp_params.z_m = cp.Parameter((N, n_z), name="z_minus")
        n_w = n_z + n_nu
        method.cp_params.L = cp.Parameter((N, n_w, n_w), name="L")

    if method.flags.discretize == "ps":
        N_col = N - 1
        _, _, _, D_np = discretize.compute_ps_differentiation_matrix(N_col)
        method.ps_D = D_np

        method.cp_params.ps_f_ref = cp.Parameter((N_col, n_z),       name="ps_f_ref")
        method.cp_params.ps_Ac    = cp.Parameter((N_col, n_z, n_z),  name="ps_Ac")
        method.cp_params.ps_Bc    = cp.Parameter((N_col, n_z, n_nu), name="ps_Bc")

def create_cvxpy_constraints_dynamics(method):

    problem = method.problem
    N       = method.N.time_grid

    # pseudo-spectral dynamics constraint
    if method.flags.discretize == "ps":
        method.cp_constraints.append(discretize.build_ps_dyn_constraints(method))

    # multiple shooting dynamics constraint
    if method.flags.discretize == "ms":
        vb_dyn = method.cp_vars.vb_stack.get("dynamics", None)
        method.cp_dyn_constraints = []

        for k in range(N - 1):
            
            z_ref_prop_kp = method.cp_params.z_m[k + 1]

            z_kp = method.dz[k + 1] + method.cp_params.z_ref[k + 1]
            dz_k   = method.dz[k]
            dnu_k  = method.dnu[k]
            dnu_kp = method.dnu[k + 1]

            Ak = method.cp_params.Ak[k]
            Bk = method.cp_params.Bk[k]
            Bkp = method.cp_params.Bkp[k]

            cnst = (z_kp == (z_ref_prop_kp + Ak @ dz_k + Bk @ dnu_k + Bkp @ dnu_kp + (vb_dyn[k] if vb_dyn is not None else 0)))
            method.cp_dyn_constraints.append(cnst)
            method.cp_constraints.append(cnst)

    # ctcs integral constraint
    if problem.constraints.has(ct=1):
        for k in range(N - 1):
            method.cp_constraints.append(
                (method.cp_params.z_ref[k + 1, method.indices.z.ctcs] + method.dz[k + 1, method.indices.z.ctcs])
                - (method.cp_params.z_ref[k, method.indices.z.ctcs] + method.dz[k, method.indices.z.ctcs])
                <= method.cp_params.eps_ctcs)

# =============================================================================
# INITIAL STATE
# =============================================================================


def create_cvxpy_constraints_initial_state(method):
    for constraint in method.problem.constraints.get(type="initial_state"):
        cnst_idx = constraint.idx
        vb_init = method.cp_vars.vb_stack.get("initial_state", None)
        if vb_init is not None:
            method.cp_constraints.append(method.dz[0, cnst_idx] + method.cp_params.z_ref[0, cnst_idx] - vb_init == constraint.value)
        else:
            method.cp_constraints.append(method.dz[0, cnst_idx] + method.cp_params.z_ref[0, cnst_idx] == constraint.value)


# =============================================================================
# FINAL STATE
# =============================================================================

def create_cvxpy_constraints_final_state(method):
    for constraint in method.problem.constraints.get(type="final_state"):
        cnst_idx = constraint.idx
        value    = constraint.value
        vb_term  = method.cp_vars.vb_stack.final_state if method.cp_vars.vb_stack.final_state is not None else 0.0

        method.cp_constraints.append(method.dz[-1, cnst_idx] + method.cp_params.z_ref[-1, cnst_idx] - vb_term == value)


# =============================================================================
# NONCONVEX INEQUALITY
# =============================================================================

def create_cvxpy_parameters_nonconvex_inequality(method):
    if not method.problem.constraints.has(ct=0, type="nonconvex_inequality"):
        return

    N, n_z, n_nu = method.N.time_grid, method.n.z, method.n.nu
    n_ineq       = method.problem.index_map.n.nonconvex_inequality

    method.cp_params.dgdz  = cp.Parameter((N, n_ineq, n_z),  name="dgdz")
    method.cp_params.dgdnu = cp.Parameter((N, n_ineq, n_nu), name="dgdnu")
    method.cp_params.g0    = cp.Parameter((N, n_ineq),        name="g0")


def create_cvxpy_constraints_nonconvex_inequality(method):
    if not method.problem.constraints.has(ct=0, type="nonconvex_inequality"):
        return

    method.cp_ineq_constraints = []
    for k in range(method.N.time_grid):
        cnst = (
            method.cp_params.dgdz[k] @ method.dz[k, :]
            + method.cp_params.dgdnu[k] @ method.dnu[k, :]
            + method.cp_params.g0[k]
            - method.cp_vars.vb_stack.nonconvex_inequality[k]
            <= 0
        )
        method.cp_ineq_constraints.append(cnst)
        method.cp_constraints.append(cnst)
        method.cp_constraints.append(method.cp_vars.vb_stack.nonconvex_inequality[k] >= 0)


# =============================================================================
# CONVEX INEQUALITY
# =============================================================================

def create_cvxpy_constraints_convex_inequality(method):
    convex_ineq_constraints = method.problem.constraints.get(ct=0, type="convex_inequality")
    if not convex_ineq_constraints:
        return

    params = method.problem.params
    state_idx = method.indices.z.state
    ctrl_idx = method.indices.nu.control

    x_all = method.cp_params.z_ref[:, state_idx] + method.dz[:, state_idx]
    u_all = method.cp_params.nu_ref[:, ctrl_idx] + method.dnu[:, ctrl_idx]

    for constraint in convex_ineq_constraints:
        nodes = constraint.nodes
        x_nodes = x_all[nodes]
        u_nodes = u_all[nodes]
        expr = constraint.fcn(x_nodes, u_nodes, params)
        method.cp_constraints.append(expr <= 0)


# =============================================================================
# STATE LIMITS
# =============================================================================

def create_cvxpy_constraints_state_limits(method):
    for k in range(method.N.time_grid):
        for constraint in method.problem.constraints.get(ct=0, type="state_limits"):
            if k in constraint.nodes:
                x_k = method.cp_params.z_ref[k, method.indices.z.state] + method.dz[k, method.indices.z.state]
                if constraint.lower_idx:
                    method.cp_constraints.append(x_k[constraint.lower_idx] >= constraint.lower_value)
                if constraint.upper_idx:
                    method.cp_constraints.append(x_k[constraint.upper_idx] <= constraint.upper_value)


# =============================================================================
# CONTROL LIMITS
# =============================================================================

def create_cvxpy_constraints_control_limits(method):
    for k in range(method.N.time_grid):
        for constraint in method.problem.constraints.get(ct=0, type="control_limits"):
            if k in constraint.nodes:
                u_k = method.cp_params.nu_ref[k, method.indices.nu.control] + method.dnu[k, method.indices.nu.control]
                if constraint.lower_idx:
                    method.cp_constraints.append(u_k[constraint.lower_idx] >= constraint.lower_value)
                if constraint.upper_idx:
                    method.cp_constraints.append(u_k[constraint.upper_idx] <= constraint.upper_value)


# =============================================================================
# CONTROL RATE LIMIT
# =============================================================================

def create_cvxpy_constraints_control_rate_limit(method):
    for k in range(method.N.time_grid - 1):
        for constraint in method.problem.constraints.get(ct=0, type="control_rate_limit"):
            value = constraint.value
            M_sel = constraint.M_select
            du_k  = (
                method.cp_params.nu_ref[k + 1, method.indices.nu.control] + method.dnu[k + 1, method.indices.nu.control]
                - (method.cp_params.nu_ref[k, method.indices.nu.control]  + method.dnu[k,     method.indices.nu.control])
            )
            dt_k = (method.t_ref[k + 1, 0] + method.dt[k + 1, 0]) - (method.t_ref[k, 0] + method.dt[k, 0])
            method.cp_constraints.append(M_sel @ du_k <= dt_k * np.concatenate([value, value]))


# =============================================================================
# FREE FINAL TIME
# =============================================================================

def create_cvxpy_parameters_free_final_time(method):
    method.cp_params.dt_interval_min = cp.Parameter(nonneg=True, name="dt_interval_min")


def create_cvxpy_constraints_free_final_time(method):
    N = method.N.time_grid

    method.cp_constraints.append(method.dt[0, 0] == 0)

    for k in range(N - 1):
        t_k = method.t_ref[k, 0] + method.dt[k, 0]

        t_kp = method.t_ref[k + 1, 0] + method.dt[k + 1, 0]

        method.cp_constraints.append(t_k >= 0)
        method.cp_constraints.append(t_kp - t_k >= method.cp_params.dt_interval_min)
        method.cp_constraints.append(0.0 <= method.s_ref[k, 0] + method.ds[k, 0])

        if hasattr(method.flags, "equal_dt") and bool(method.flags.equal_dt) and k >= 1:
            interval_k = (method.t_ref[k + 1, 0] + method.dt[k + 1, 0]) - (method.t_ref[k, 0] + method.dt[k, 0])
            interval_0 = (method.t_ref[1, 0] + method.dt[1, 0]) - (method.t_ref[0, 0] + method.dt[0, 0])
            method.cp_constraints.append(interval_k == interval_0)

        if hasattr(method.flags, "zoh_dilation") and bool(method.flags.zoh_dilation):
            s_k  = method.s_ref[k, 0] + method.ds[k, 0]
            s_kp = method.s_ref[k + 1, 0] + method.ds[k + 1, 0]
            method.cp_constraints.append(s_k == s_kp)

    method.cp_constraints.append(0.0 <= method.s_ref[N - 1, 0] + method.ds[N - 1, 0])


# =============================================================================
# FINAL TIME
# =============================================================================

def create_cvxpy_parameters_final_time(method):
    method.cp_params.T_min   = cp.Parameter(nonneg=True, name="T_min")
    method.cp_params.T_max   = cp.Parameter(nonneg=True, name="T_max")
    method.cp_params.dt_min  = cp.Parameter(nonneg=True, name="dt_min")
    method.cp_params.dt_max  = cp.Parameter(nonneg=True, name="dt_max")


def create_cvxpy_constraints_final_time(method):
    N = method.N.time_grid

    method.cp_constraints.append(method.dt[0, 0] == 0)

    for k in range(N - 1):
        t_k  = method.t_ref[k,     0] + method.dt[k,     0]
        t_kp = method.t_ref[k + 1, 0] + method.dt[k + 1, 0]
        t_interval_k = t_kp - t_k

        method.cp_constraints.append(t_interval_k <= method.cp_params.dt_max)
        method.cp_constraints.append(t_interval_k >= method.cp_params.dt_min)

    method.cp_constraints.append(method.cp_params.T_min <= method.t_ref[-1, 0] + method.dt[-1, 0])
    method.cp_constraints.append(method.t_ref[-1, 0] + method.dt[-1, 0] <= method.cp_params.T_max)


def update_cvxpy_parameters_final_time_values(method):
    for constraint in method.problem.constraints.get(type="final_time"):
        if constraint.lower is not None:
            method.cp_params.T_min.value  = float(constraint.lower)
            method.cp_params.dt_min.value = float(constraint.dt_min)
        if constraint.upper is not None:
            method.cp_params.T_max.value  = float(constraint.upper)
            method.cp_params.dt_max.value = float(constraint.dt_max)


# =============================================================================
# AXIS ANGLE CONE
# =============================================================================

def create_cvxpy_constraints_axis_angle_cone(method):
    for k in range(method.N.time_grid):
        for constraint in method.problem.constraints.get(ct=0, type="axis_angle_cone"):
            cnst_idx = constraint.idx
            axis     = constraint.axis
            if constraint.set == "state":
                method.cp_constraints.append(
                    constraint.cos_theta_max * cp.norm(method.cp_params.z_ref[k, cnst_idx] + method.dz[k, cnst_idx])
                    <= axis @ (method.cp_params.z_ref[k, cnst_idx] + method.dz[k, cnst_idx])
                )
            elif constraint.set == "control":
                method.cp_constraints.append(
                    constraint.cos_theta_max * cp.norm(method.cp_params.nu_ref[k, cnst_idx] + method.dnu[k, cnst_idx])
                    <= axis @ (method.cp_params.nu_ref[k, cnst_idx] + method.dnu[k, cnst_idx])
                )


# =============================================================================
# MAX NORM CONE
# =============================================================================

def create_cvxpy_constraints_max_norm_cone(method):
    for k in range(method.N.time_grid):
        for constraint in method.problem.constraints.get(ct=0, type="max_norm_cone"):
            cnst_idx  = constraint.idx
            upper = constraint.upper
            if constraint.set == "state":
                method.cp_constraints.append(cp.norm(method.cp_params.z_ref[k, cnst_idx] + method.dz[k, cnst_idx]) <= upper)
            elif constraint.set == "control":
                method.cp_constraints.append(cp.norm(method.cp_params.nu_ref[k, cnst_idx] + method.dnu[k, cnst_idx]) <= upper)


# =============================================================================
# QUATERNION CONE
# =============================================================================

def create_cvxpy_constraints_quaternion_cone(method):
    for k in range(method.N.time_grid):
        for constraint in method.problem.constraints.get(ct=0, type="quaternion_cone"):
            q_idx = constraint.quat_start_idx
            method.cp_constraints.append(
                cp.norm(
                    method.cp_params.z_ref[k, q_idx + 2 : q_idx + 4] + method.dz[k, q_idx + 2 : q_idx + 4]
                ) <= constraint.rhs
            )


# =============================================================================
# AFFINE
# =============================================================================

def create_cvxpy_constraints_AFFINE(method):
    N = method.N.time_grid
    for k in range(N):
        for constraint in method.problem.constraints.get(ct=0, type="AFFINE"):
            if not constraint.convex:
                continue
            if not _active_at(k, N, constraint.time_steps):
                continue
            cnst_idx = constraint.idx
            AA, bb   = constraint.A, constraint.b
            if constraint.set == "state":
                method.cp_constraints.append(AA @ (method.cp_params.z_ref[k, cnst_idx]  + method.dz[k,  cnst_idx]) == bb)
            elif constraint.set == "control":
                method.cp_constraints.append(AA @ (method.cp_params.nu_ref[k, cnst_idx] + method.dnu[k, cnst_idx]) == bb)


# =============================================================================
# POLYTOPE
# =============================================================================

def create_cvxpy_constraints_POLYTOPE(method):
    N = method.N.time_grid
    for k in range(N):
        for constraint in method.problem.constraints.get(ct=0, type="POLYTOPE"):
            if not constraint.convex:
                continue
            if not _active_at(k, N, constraint.time_steps):
                continue
            cnst_idx = constraint.idx
            AA, bb   = constraint.A, constraint.b
            if constraint.set == "state":
                method.cp_constraints.append(AA @ (method.cp_params.z_ref[k, cnst_idx]  + method.dz[k,  cnst_idx]) <= bb)
            elif constraint.set == "control":
                method.cp_constraints.append(AA @ (method.cp_params.nu_ref[k, cnst_idx] + method.dnu[k, cnst_idx]) <= bb)


# =============================================================================
# SOC
# =============================================================================

def create_cvxpy_constraints_SOC(method):
    N = method.N.time_grid
    for k in range(N):
        for constraint in method.problem.constraints.get(ct=0, type="SOC"):
            if not constraint.convex:
                continue
            if not _active_at(k, N, constraint.time_steps):
                continue
            cnst_idx     = constraint.idx
            AA, bb, CC, dd = constraint.A, constraint.b, constraint.C, constraint.d
            if constraint.set == "state":
                method.cp_constraints.append(
                    cp.norm(AA @ (method.cp_params.z_ref[k, cnst_idx] + method.dz[k, cnst_idx]) + bb)
                    <= CC @ (method.cp_params.z_ref[k, cnst_idx] + method.dz[k, cnst_idx]) + dd
                )
            elif constraint.set == "control":
                method.cp_constraints.append(
                    cp.norm(AA @ (method.cp_params.nu_ref[k, cnst_idx] + method.dnu[k, cnst_idx]) + bb)
                    <= CC @ (method.cp_params.nu_ref[k, cnst_idx] + method.dnu[k, cnst_idx]) + dd
                )


# =============================================================================
# MINIMAX EPIGRAPH
# =============================================================================

def create_cvxpy_variables_minimax_epigraph(method):
    if method.problem.costs.has(type="nonconvex", minimax=1):
        method.minimax_epigraph_upperbound = cp.Variable((1,), name="minimax_epigraph_upperbound")


def create_cvxpy_constraints_minimax_epigraph(method):
    for k in range(method.N.time_grid):
        for cost in method.problem.costs.get(type="nonconvex", minimax=1):
            method.cp_constraints.append(
                method.cp_params.w_cost_times_dcostdx[k] @ method.dz[k, :]
                + method.cp_params.w_cost_times_dcostdu[k] @ method.dnu[k, :]
                + method.cp_params.w_cost_times_cost0[k]
                <= method.minimax_epigraph_upperbound
            )


# =============================================================================
# Utility
# =============================================================================

def _active_at(k, N, time_steps):
    if time_steps == "all":
        return True
    return k in time_steps or (k - N) in time_steps


# =============================================================================
# COSTS
# Each function receives `method` and accumulates into method.cp_cost.
# =============================================================================

def create_cvxpy_cost_nonconvex(method):
    JAX_TYPES = ("nonconvex", "terminal", "running")
    jax_costs = [c for t in JAX_TYPES for c in method.problem.costs.get(type=t)]
    if not jax_costs:
        return
    first_jax_type = method.problem.costs.cost_type_list[
        next(i for i, t in enumerate(method.problem.costs.cost_type_list) if t in JAX_TYPES)
    ]
    if first_jax_type != "nonconvex":
        return
    method.cp_cost += (
        cp.sum(method.cp_params.w_cost_times_cost0)
        + cp.sum(cp.multiply(method.cp_params.w_cost_times_dcostdx, method.dz))
        + cp.sum(cp.multiply(method.cp_params.w_cost_times_dcostdu, method.dnu))
    )


def create_cvxpy_cost_terminal(method):
    JAX_TYPES = ("nonconvex", "terminal", "running")
    jax_costs = [c for t in JAX_TYPES for c in method.problem.costs.get(type=t)]
    if not jax_costs:
        return
    first_jax_type = method.problem.costs.cost_type_list[
        next(i for i, t in enumerate(method.problem.costs.cost_type_list) if t in JAX_TYPES)
    ]
    if first_jax_type != "terminal":
        return
    method.cp_cost += (
        cp.sum(method.cp_params.w_cost_times_cost0)
        + cp.sum(cp.multiply(method.cp_params.w_cost_times_dcostdx, method.dz))
        + cp.sum(cp.multiply(method.cp_params.w_cost_times_dcostdu, method.dnu))
    )


def create_cvxpy_cost_running(method):
    JAX_TYPES = ("nonconvex", "terminal", "running")
    jax_costs = [c for t in JAX_TYPES for c in method.problem.costs.get(type=t)]
    if not jax_costs:
        return
    first_jax_type = method.problem.costs.cost_type_list[
        next(i for i, t in enumerate(method.problem.costs.cost_type_list) if t in JAX_TYPES)
    ]
    if first_jax_type != "running":
        return
    method.cp_cost += (
        cp.sum(method.cp_params.w_cost_times_cost0)
        + cp.sum(cp.multiply(method.cp_params.w_cost_times_dcostdx, method.dz))
        + cp.sum(cp.multiply(method.cp_params.w_cost_times_dcostdu, method.dnu))
    )


def create_cvxpy_cost_convex_terminal(method):
    for cost in method.problem.costs.get(type="convex_terminal"):
        for k in np.atleast_1d(cost.nodes):
            x_k = method.cp_params.z_ref[k, method.indices.z.state] + method.dz[k, method.indices.z.state]
            u_k = method.cp_params.nu_ref[k, method.indices.nu.control] + method.dnu[k, method.indices.nu.control]
            method.cp_cost += cost.w * cost.fcn_dim(0, x_k, u_k, method.problem.params)


def create_cvxpy_cost_convex_running(method):
    N = method.N.time_grid
    for cost in method.problem.costs.get(type="convex_running"):
        for k in range(N):
            if k in cost.nodes:
                x_k = method.cp_params.z_ref[k, method.indices.z.state] + method.dz[k, method.indices.z.state]
                u_k = method.cp_params.nu_ref[k, method.indices.nu.control] + method.dnu[k, method.indices.nu.control]
                method.cp_cost += cost.w * cost.fcn_dim(0, x_k, u_k, method.problem.params)


def create_cvxpy_cost_minimax_epigraph(method):
    for cost in method.problem.costs.get(type="nonconvex", minimax=1):
        method.cp_cost += method.minimax_epigraph_upperbound


def create_cvxpy_cost_min_time(method):
    if bool(method.flags.free_final_time):
        for cost in method.problem.costs.get(type="min_time"):
            method.cp_cost += cp.sum(method.t_ref + method.dt)


def create_cvxpy_cost_min_norm_terminal(method):
    for cost in method.problem.costs.get(type="min_norm_terminal"):
        zf     = method.cp_params.z_ref[-1] + method.dz[-1]
        target = cost.value if cost.value is not None else np.zeros(len(cost.idx))
        method.cp_cost += method.cp_params.w_cost * cp.norm(zf[cost.idx] - target)


def create_cvxpy_cost_terminal_state(method):
    for cost in method.problem.costs.get(type="terminal_state"):
        zf = method.cp_params.z_ref[-1] + method.dz[-1]
        method.cp_cost += cost.sign * method.cp_params.w_cost * zf[cost.idx]


def create_cvxpy_cost_regularization(method):
    for cost in method.problem.costs.get(type="regularization"):
        if cost.set == "control":
            traj = method.cp_params.nu_ref + method.dnu
        elif cost.set == "state":
            traj = method.cp_params.z_ref + method.dz
        else:
            continue

        if cost.norm_type == "l2":
            method.cp_cost += cost.w * cp.sum_squares(traj)
        elif cost.norm_type == "l1":
            method.cp_cost += cost.w * cp.norm1(traj)


def create_cvxpy_cost_rate_regularization(method):
    for cost in method.problem.costs.get(type="rate_regularization"):
        if cost.set == "control":
            traj = method.cp_params.nu_ref + method.dnu
        elif cost.set == "state":
            traj = method.cp_params.z_ref + method.dz
        else:
            continue

        traj_minus = traj[:-1, cost.idx]
        traj_plus  = traj[1:,  cost.idx]
        delta      = traj_plus - traj_minus

        if cost.norm_type == "l2":
            method.cp_cost += cost.w * cp.sum_squares(delta)
        elif cost.norm_type == "l1":
            method.cp_cost += cost.w * cp.norm1(delta)


def create_cvxpy_cost_trust_region(method):
    N = method.N.time_grid
    for k in range(N):
        w_k = cp.hstack([method.dz[k], method.dnu[k]])
        method.cp_cost += 0.5 * cp.sum_squares(method.cp_params.L[k] @ w_k)
    # method.cp_cost += 0.001 * (cp.sum_squares(method.dz) + cp.sum_squares(method.dnu))

def create_cvxpy_cost_quadratic_penalty(method):
    VB_quad = 0.0
    for cnstr_type in method.W_stack.keys():
        vb = method.cp_vars.vb_stack.get(cnstr_type)
        if vb is None:
            continue

        W_sqrt = method.cp_params.W_sqrt.get(cnstr_type)
        if W_sqrt is None:
            continue
        if hasattr(W_sqrt, "shape") and len(W_sqrt.shape) == 2:
            for k in range(W_sqrt.shape[0]):
                VB_quad += cp.sum_squares(cp.diag(W_sqrt[k, :]) @ vb[k, :])
        else:
            VB_quad += cp.sum_squares(cp.diag(W_sqrt) @ vb)
    method.cp_cost += 0.5 * VB_quad


def create_cvxpy_cost_dual(method):
    DUAL = 0.0
    for cnstr_type in method.dual_stack.keys():
        dual_param = method.cp_params.dual.get(cnstr_type)
        vb         = method.cp_vars.vb_stack.get(cnstr_type)
        if dual_param is None or vb is None:
            continue
        DUAL += cp.sum(cp.multiply(vb, dual_param))
    method.cp_cost += DUAL




def _psd_sqrt(H_batch):
    eigvals, eigvecs = np.linalg.eigh(H_batch)
    sqrt_eigvals = np.sqrt(np.maximum(eigvals, 1e-6))
    return sqrt_eigvals[..., :, np.newaxis] * np.transpose(eigvecs, (0, 2, 1))


# =============================================================================
# MERIT FUNCTION & LINE SEARCH
# =============================================================================

def compile_merit_function(method):
    """Build JIT-compiled augmented Lagrangian merit function and its
    alpha-gradient for Armijo line search.

    phi(z,nu) = objective + sum_types[ dual^T viol + 0.5 W * viol^2 ]
    """
    propagate = method.propagate
    ks = jnp.arange(method.N.time_grid - 1)
    N_grid = method.N.time_grid
    w_cost = float(method.config.method.weights.w_cost)

    indices = method.index_map.indices

    # --- Collect all cost contributions ---
    running_cost_fns = [
        cost_fn.fcn_batched
        for cost_fn in (
            method.problem.costs.get(type="nonconvex")
            + method.problem.costs.get(type="running")
        )
    ]
    terminal_cost_fns = [
        (cost_fn.fcn_batched, jnp.asarray(cost_fn.nodes))
        for cost_fn in method.problem.costs.get(type="terminal")
    ]

    # terminal_state costs: sign * z[-1, idx]
    terminal_state_info = [
        (cost_fn.sign, jnp.array(cost_fn.idx))
        for cost_fn in method.problem.costs.get(type="terminal_state")
    ]

    # min_time costs: sum of time components
    has_min_time = len(method.problem.costs.get(type="min_time")) > 0
    time_idx = jnp.array(indices.z.time) if has_min_time else None

    # regularization costs: w * ||traj||^2 or w * ||traj||_1
    reg_info = []
    for cost_fn in method.problem.costs.get(type="regularization"):
        is_nu = (cost_fn.set == "control")
        reg_info.append((cost_fn.w, cost_fn.norm_type, is_nu))

    # rate_regularization costs: w * ||diff(traj)||^2 or w * ||diff(traj)||_1
    rate_reg_info = []
    for cost_fn in method.problem.costs.get(type="rate_regularization"):
        is_nu = (cost_fn.set == "control")
        rate_reg_info.append((cost_fn.w, cost_fn.norm_type, is_nu))

    # --- Penalty setup ---
    penalty_types = set(method.W_stack.keys()) | set(method.dual_stack.keys())

    has_dyn       = "dynamics" in penalty_types
    has_dyn_W     = "dynamics" in method.W_stack
    has_dyn_dual  = "dynamics" in method.dual_stack

    has_ineq      = "nonconvex_inequality" in penalty_types
    has_ineq_W    = "nonconvex_inequality" in method.W_stack
    has_ineq_dual = "nonconvex_inequality" in method.dual_stack
    ineq_info = []
    n_ineq = 0
    if has_ineq and hasattr(method.n, "nonconvex_inequality"):
        n_ineq = method.n.nonconvex_inequality
        col = 0
        if method.problem.constraints.has(ct=0, type="nonconvex_inequality"):
            for c in method.problem.constraints.get(ct=0, type="nonconvex_inequality"):
                ineq_info.append((c.fcn_batched, jnp.asarray(c.nodes), col, c.dimension))
                col += c.dimension

    has_final      = "final_state" in penalty_types
    has_final_W    = "final_state" in method.W_stack
    has_final_dual = "final_state" in method.dual_stack
    final_info = []
    n_final = 0
    if has_final and hasattr(method.n, "final_state"):
        n_final = method.n.final_state
        col = 0
        for c in method.problem.constraints.get(type="final_state"):
            dim_c = len(c.idx)
            final_info.append((jnp.array(c.idx), jnp.asarray(c.value), col, dim_c))
            col += dim_c

    def merit_fn(z, nu, W_dict, dual_dict, params):
        obj = 0.0

        # nonconvex / running / terminal costs (JAX-based)
        for fcn_b in running_cost_fns:
            obj = obj + jnp.sum(fcn_b(z, nu, params))
        for fcn_b, nodes in terminal_cost_fns:
            obj = obj + jnp.sum(fcn_b(z[nodes], nu[nodes], params))

        # terminal_state costs: sign * sum(z[-1, idx])
        for sign, idx in terminal_state_info:
            obj = obj + sign * jnp.sum(z[-1, idx])

        # min_time cost: sum of all time values
        if has_min_time:
            obj = obj + jnp.sum(z[:, time_idx])

        # regularization costs
        for w, norm_type, is_nu in reg_info:
            traj = nu if is_nu else z
            if norm_type == "l2":
                obj = obj + w * jnp.sum(traj ** 2)
            else:
                obj = obj + w * jnp.sum(jnp.abs(traj))

        # rate regularization costs
        for w, norm_type, is_nu in rate_reg_info:
            traj = nu if is_nu else z
            delta = traj[1:] - traj[:-1]
            if norm_type == "l2":
                obj = obj + w * jnp.sum(delta ** 2)
            else:
                obj = obj + w * jnp.sum(jnp.abs(delta))

        obj = w_cost * obj

        al = 0.0

        if has_dyn:
            z_minus = propagate(ks, z[:-1], nu[:-1], nu[1:], params)
            viol = z[1:] - z_minus
            if has_dyn_dual:
                al = al + jnp.sum(dual_dict["dynamics"] * viol)
            if has_dyn_W:
                al = al + 0.5 * jnp.sum(W_dict["dynamics"] * viol ** 2)

        if has_ineq and n_ineq > 0:
            viol_ineq = jnp.zeros((N_grid, n_ineq))
            for fcn_b, nodes, cs, dc in ineq_info:
                g = jnp.maximum(0.0, fcn_b(z[nodes], nu[nodes], params))
                viol_ineq = viol_ineq.at[nodes, cs:cs + dc].set(g.reshape(-1, dc))
            if has_ineq_dual:
                al = al + jnp.sum(dual_dict["nonconvex_inequality"] * viol_ineq)
            if has_ineq_W:
                al = al + 0.5 * jnp.sum(W_dict["nonconvex_inequality"] * viol_ineq ** 2)

        if has_final and n_final > 0:
            viol_f = jnp.zeros((1, n_final))
            for idx, val, cs, dc in final_info:
                viol_f = viol_f.at[0, cs:cs + dc].set(z[-1, idx] - val)
            if has_final_dual:
                al = al + jnp.sum(dual_dict["final_state"] * viol_f)
            if has_final_W:
                al = al + 0.5 * jnp.sum(W_dict["final_state"] * viol_f ** 2)

        return obj + al

    def merit_along_line(alpha, z_ref, dz, nu_ref, dnu, W_dict, dual_dict, params):
        return merit_fn(z_ref + alpha * dz, nu_ref + alpha * dnu, W_dict, dual_dict, params)

    method._merit_fn_jax = jax.jit(merit_fn)
    method._merit_value_and_grad_alpha = jax.jit(
        jax.value_and_grad(merit_along_line, argnums=0)
    )
    method._phi_history = []


def armijo_line_search(method, dz, dnu, c1=1e-4, beta=0.5, max_ls_iter=20, alpha_min=0.2, m=5):
    z_ref   = jnp.asarray(method.current_iter_data.z_opt)
    nu_ref  = jnp.asarray(method.current_iter_data.nu_opt)
    dz_jax  = jnp.asarray(dz)
    dnu_jax = jnp.asarray(dnu)
    W_dict    = {t: jnp.asarray(v) for t, v in method.W_stack.items()}
    dual_dict = {t: jnp.asarray(v) for t, v in method.dual_stack.items()}
    params    = method.problem.params

    phi_0, dphi = method._merit_value_and_grad_alpha(0.0, z_ref, dz_jax, nu_ref, dnu_jax, W_dict, dual_dict, params)
    phi_0, dphi = float(phi_0), float(dphi)

    phi_ref = max(method._phi_history[-m:]) if method._phi_history else phi_0
    method._phi_history.append(phi_0)

    slope = min(dphi, -abs(dphi) * 1e-6)

    alpha = 1.0
    for _ in range(max_ls_iter):
        phi_trial = float(method._merit_fn_jax(
            z_ref + alpha * dz_jax, nu_ref + alpha * dnu_jax, W_dict, dual_dict, params
        ))
        if np.isfinite(phi_trial) and phi_trial <= phi_ref + c1 * alpha * slope:
            return alpha
        alpha *= beta
        if alpha < alpha_min:
            return alpha_min

    return alpha_min


def update_cvxpy_parameters_dynamics(method):
    z_opt  = method.current_iter_data.z_opt
    nu_opt = method.current_iter_data.nu_opt

    if method.flags.discretize == "ms":
        Ak, Bk, Bkp, z_minus = discretize.compute_linsys_discrete(z_opt, nu_opt, method.problem, method)
        method.cp_params.Ak.value  = Ak
        method.cp_params.Bk.value  = Bk
        method.cp_params.Bkp.value = Bkp
        method.cp_params.z_m.value = z_minus

        N, n_z, n_nu = method.N.time_grid, method.n.z, method.n.nu
        n_w = n_z + n_nu

        H = np.zeros((N, n_w, n_w))

        H[:-1, :n_z, :n_z] += np.asarray(method.H_z_k[0])
        H[:-1, :n_z, n_z:] += np.asarray(method.H_z_k[1])
        H[:-1, n_z:, :n_z] += np.asarray(method.H_nu_k[0])
        H[:-1, n_z:, n_z:] += np.asarray(method.H_nu_k[1])

        H[1:, n_z:, n_z:] += np.asarray(method.H_nu_kp[2])

        H_cost_z, H_cost_nu, H_cost_znu = discretize.compute_nonconvex_cost_hessians(z_opt, nu_opt, method.problem, method)
        w_cost = method.cp_params.w_cost.value
        
        H[:, :n_z, :n_z] += w_cost * H_cost_z
        H[:, n_z:, n_z:] += w_cost * H_cost_nu
        H[:, :n_z, n_z:] += w_cost * H_cost_znu
        H[:, n_z:, :n_z] += w_cost * np.transpose(H_cost_znu, (0, 2, 1))

        H_ineq = discretize.compute_nonconvex_constraint_hessians(z_opt, nu_opt, method.problem, method)
        H += H_ineq

        method.cp_params.L.value = _psd_sqrt(H)
        # method.cp_params.L.value += 0.001 * np.eye(n_w)[np.newaxis, :, :]

    if method.flags.discretize == "ps":
        f_ref_col, Ac_col, Bc_col = discretize.compute_ps_dynamics_and_jacobians(z_opt, nu_opt, method.problem, method)
        method.cp_params.ps_f_ref.value = f_ref_col
        method.cp_params.ps_Ac.value    = Ac_col
        method.cp_params.ps_Bc.value    = Bc_col


def update_cvxpy_parameters_nonconvex_inequality(method):
    z_opt  = method.current_iter_data.z_opt
    nu_opt = method.current_iter_data.nu_opt

    if method.problem.constraints.has(ct=0, type="nonconvex_inequality"):
        g, dgdz, dgdnu = discretize.compute_nonconvex_constraints(z_opt, nu_opt, method.problem, method)
        
        method.cp_params.dgdz.value  = dgdz
        method.cp_params.dgdnu.value = dgdnu
        method.cp_params.g0.value    = g


def update_cvxpy_parameters_nonconvex_costs(method):

    z_opt  = method.current_iter_data.z_opt
    nu_opt = method.current_iter_data.nu_opt

    cost, dcostdx, dcostdu = discretize.compute_nonconvex_costs(
        z_opt, nu_opt, method.problem, method
    )
    method.cp_params.w_cost_times_dcostdx.value = method.cp_params.w_cost.value * dcostdx.squeeze(axis=1)
    method.cp_params.w_cost_times_dcostdu.value = method.cp_params.w_cost.value * dcostdu.squeeze(axis=1)
    method.cp_params.w_cost_times_cost0.value   = method.cp_params.w_cost.value * cost.squeeze(axis=1)


def update_cvxpy_parameters_free_final_time_(method):
    if bool(method.flags.free_final_time):
        method.cp_params.dt_interval_min.value = 0.2 * method.dt_init_min


# =============================================================================
# UPDATE CURRENT ITER DATA
# =============================================================================

def update_current_iter_data_dynamics(method):
    z_opt  = method.current_iter_data.z_opt
    nu_opt = method.current_iter_data.nu_opt

    if method.flags.discretize == "ms":
        ks         = jnp.arange(method.index_map.N.time_grid - 1)
        z_ref_ks   = jnp.asarray(z_opt[:-1])
        nu_ref_ks  = jnp.asarray(nu_opt[:-1])
        nu_ref_kps = jnp.asarray(nu_opt[1:])
        z_minus    = np.asarray(method.propagate(ks, z_ref_ks, nu_ref_ks, nu_ref_kps, method.problem.params))
        method.current_iter_data.defect = z_opt[1:] - z_minus


def update_current_iter_data_nonconvex_inequality(method):
    problem = method.problem
    z_opt  = method.current_iter_data.z_opt
    nu_opt = method.current_iter_data.nu_opt
    g_nl, _, _ = discretize.compute_nonconvex_constraints(z_opt, nu_opt, problem, method)
    method.current_iter_data.g_nonconvex_inequality = g_nl


# =============================================================================
# REGISTRIES
# =============================================================================

CREATE_PARAMETER_REGISTRY.update({
    "dynamics":               create_cvxpy_parameters_dynamics,
    "nonconvex_inequality":   create_cvxpy_parameters_nonconvex_inequality,
    "final_time":             create_cvxpy_parameters_final_time,
    "free_final_time":        create_cvxpy_parameters_free_final_time,
})

CREATE_VARIABLE_REGISTRY.update({
    "minimax_epigraph":       create_cvxpy_variables_minimax_epigraph,
})

CREATE_CONSTRAINT_REGISTRY.update({
    "dynamics":               create_cvxpy_constraints_dynamics,
    "initial_state":          create_cvxpy_constraints_initial_state,
    "final_state":            create_cvxpy_constraints_final_state,
    "nonconvex_inequality":   create_cvxpy_constraints_nonconvex_inequality,
    "convex_inequality":      create_cvxpy_constraints_convex_inequality,
    "state_limits":           create_cvxpy_constraints_state_limits,
    "control_limits":         create_cvxpy_constraints_control_limits,
    "control_rate_limit":     create_cvxpy_constraints_control_rate_limit,
    "final_time":             create_cvxpy_constraints_final_time,
    "free_final_time":        create_cvxpy_constraints_free_final_time,
    "axis_angle_cone":        create_cvxpy_constraints_axis_angle_cone,
    "max_norm_cone":          create_cvxpy_constraints_max_norm_cone,
    "quaternion_cone":        create_cvxpy_constraints_quaternion_cone,
    "AFFINE":                 create_cvxpy_constraints_AFFINE,
    "POLYTOPE":               create_cvxpy_constraints_POLYTOPE,
    "SOC":                    create_cvxpy_constraints_SOC,
    "minimax_epigraph":       create_cvxpy_constraints_minimax_epigraph,
})

CREATE_COST_REGISTRY.update({
    "nonconvex":              create_cvxpy_cost_nonconvex,
    "terminal":               create_cvxpy_cost_terminal,
    "running":                create_cvxpy_cost_running,
    "convex_terminal":        create_cvxpy_cost_convex_terminal,
    "convex_running":         create_cvxpy_cost_convex_running,
    "minimax_epigraph":       create_cvxpy_cost_minimax_epigraph,
    "min_time":               create_cvxpy_cost_min_time,
    "min_norm_terminal":      create_cvxpy_cost_min_norm_terminal,
    "terminal_state":         create_cvxpy_cost_terminal_state,
    "regularization":         create_cvxpy_cost_regularization,
    "rate_regularization":    create_cvxpy_cost_rate_regularization,
    "trust_region":           create_cvxpy_cost_trust_region,
    "quadratic_penalty":      create_cvxpy_cost_quadratic_penalty,
    "dual":                   create_cvxpy_cost_dual,
})

UPDATE_PARAMETER_REGISTRY.update({
    "dynamics":               update_cvxpy_parameters_dynamics,
    "nonconvex_inequality":   update_cvxpy_parameters_nonconvex_inequality,
    "costs":                  update_cvxpy_parameters_nonconvex_costs,
    "final_time":             update_cvxpy_parameters_final_time_values,
    "free_final_time":        update_cvxpy_parameters_free_final_time_,
})

UPDATE_CURRENT_ITER_DATA_REGISTRY.update({
    "dynamics":               update_current_iter_data_dynamics,
    "nonconvex_inequality":   update_current_iter_data_nonconvex_inequality,
})

INITIALIZE_METHOD_REGISTRY.update({
    "merit_function":         compile_merit_function,
})