from typing import TYPE_CHECKING

import cvxpy as cp
import diffrax
import jax
import jax.numpy as jnp
import numpy as np

from trajopt.problem import Problem
from trajopt.methods.scp import pseudospectral

if TYPE_CHECKING:
    from trajopt.methods.scp.scvx import SCvx

jax.config.update("jax_enable_x64", True)

def compute_nonconvex_constraints(
    z: np.ndarray, nu: np.ndarray, problem: Problem, method: "SCvx",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate nonconvex inequality constraints and their Jacobians.

    Returns:
        Tuple of (g, dgdz, dgdnu) over the time grid.

    """
    n_ineq    = problem.index_map.n.nonconvex_inequality
    n_z       = problem.index_map.n.z
    n_nu      = problem.index_map.n.nu
    N         = method.index_map.N.time_grid
    z_jax     = jnp.asarray(z)
    nu_jax    = jnp.asarray(nu)
    params = problem.params

    g     = np.zeros((N, n_ineq))
    dgdz  = np.zeros((N, n_ineq, n_z))
    dgdnu = np.zeros((N, n_ineq, n_nu))

    col_start = 0
    for constraint in problem.constraints.get(ct=0, type="nonconvex_inequality"):
        col_end = col_start + constraint.dimension
        nodes = np.asarray(constraint.nodes)

        f_batch, dfdx_batch, dfdu_batch = constraint.g_aff_batched(z_jax[nodes], nu_jax[nodes], params)

        g[nodes, col_start:col_end]      = np.asarray(f_batch)
        dgdz[nodes, col_start:col_end, :] = np.asarray(dfdx_batch)
        dgdnu[nodes, col_start:col_end, :] = np.asarray(dfdu_batch)

        col_start = col_end

    return g, dgdz, dgdnu


def compute_nonconvex_constraint_hessians(z, nu, problem, method):
    N = method.index_map.N.time_grid
    n_z = problem.index_map.n.z
    n_nu = problem.index_map.n.nu
    n_w = n_z + n_nu

    H_ineq = np.zeros((N, n_w, n_w))

    if not problem.constraints.has(ct=0, type="nonconvex_inequality"):
        return H_ineq

    lam = jnp.asarray(method.lagrangian_duals.nonconvex_inequality)
    z_jax = jnp.asarray(z)
    nu_jax = jnp.asarray(nu)
    params = problem.params

    col_start = 0
    for constraint in problem.constraints.get(ct=0, type="nonconvex_inequality"):
        col_end = col_start + constraint.dimension
        nodes = np.asarray(constraint.nodes)

        lam_nodes = lam[nodes, col_start:col_end]
        z_nodes = z_jax[nodes]
        nu_nodes = nu_jax[nodes]

        H_z, H_nu = constraint.lagrangian_hessians(lam_nodes, z_nodes, nu_nodes, params)

        for i, k in enumerate(nodes):
            H_ineq[k, :n_z, :n_z] += np.asarray(H_z[0][i])
            H_ineq[k, :n_z, n_z:] += np.asarray(H_z[1][i])
            H_ineq[k, n_z:, :n_z] += np.asarray(H_nu[0][i])
            H_ineq[k, n_z:, n_z:] += np.asarray(H_nu[1][i])

        col_start = col_end

    return H_ineq


def compute_nonconvex_costs(z, nu, problem, method):
    N         = method.index_map.N.time_grid
    n_z       = problem.index_map.n.z
    n_nu       = problem.index_map.n.nu

    cost     = np.zeros((N, 1))
    dcostdz  = np.zeros((N, 1, n_z))
    dcostdnu = np.zeros((N, 1, n_nu))

    params = problem.params
    nonconvex_costs = problem.costs.get(type="nonconvex")
    terminal_costs  = problem.costs.get(type="terminal")
    running_costs   = problem.costs.get(type="running")

    if len(nonconvex_costs) + len(terminal_costs) + len(running_costs) == 0:
        return cost, dcostdz, dcostdnu

    z_jax  = jnp.asarray(z)
    nu_jax = jnp.asarray(nu)

    for cost_fn in nonconvex_costs + running_costs:
        w = getattr(cost_fn, 'w', 1.0)
        f_batch, dfdx_batch, dfdu_batch = cost_fn.g_aff_batched(z_jax, nu_jax, params)

        f_np    = np.asarray(f_batch).reshape(N, -1)
        dfdx_np = np.asarray(dfdx_batch).reshape(N, -1, n_z)
        dfdu_np = np.asarray(dfdu_batch).reshape(N, -1, n_nu)

        cost[:, 0] += w * np.sum(f_np, axis=1)
        dcostdz[:, 0, :] += w * np.sum(dfdx_np, axis=1)
        dcostdnu[:, 0, :] += w * np.sum(dfdu_np, axis=1)

    for cost_fn in terminal_costs:
        nodes = np.atleast_1d(cost_fn.nodes)
        z_nodes  = z_jax[nodes]
        nu_nodes = nu_jax[nodes]
        f_batch, dfdx_batch, dfdu_batch = cost_fn.g_aff_batched(z_nodes, nu_nodes, params)

        f_np    = np.asarray(f_batch).reshape(len(nodes), -1)
        dfdx_np = np.asarray(dfdx_batch).reshape(len(nodes), -1, n_z)
        dfdu_np = np.asarray(dfdu_batch).reshape(len(nodes), -1, n_nu)

        for i, k in enumerate(nodes):
            cost[k, 0]       += np.sum(f_np[i])
            dcostdz[k, 0, :]  += np.sum(dfdx_np[i], axis=0)
            dcostdnu[k, 0, :] += np.sum(dfdu_np[i], axis=0)

    return cost, dcostdz, dcostdnu


def compute_nonconvex_cost_hessians(z, nu, problem, method):
    N = method.index_map.N.time_grid
    n_z = problem.index_map.n.z
    n_nu = problem.index_map.n.nu

    H_cost_z = np.zeros((N, n_z, n_z))
    H_cost_nu = np.zeros((N, n_nu, n_nu))
    H_cost_znu = np.zeros((N, n_z, n_nu))

    params = problem.params
    nonconvex_costs = problem.costs.get(type="nonconvex")
    terminal_costs = problem.costs.get(type="terminal")
    running_costs = problem.costs.get(type="running")

    if len(nonconvex_costs) + len(terminal_costs) + len(running_costs) == 0:
        return H_cost_z, H_cost_nu, H_cost_znu

    z_jax = jnp.asarray(z)
    nu_jax = jnp.asarray(nu)

    for cost_fn in nonconvex_costs + running_costs:
        w = getattr(cost_fn, 'w', 1.0)
        d2z = cost_fn.d2fcn_dz2_batched(z_jax, nu_jax, params)
        d2nu = cost_fn.d2fcn_dnu2_batched(z_jax, nu_jax, params)
        d2znu = cost_fn.d2fcn_dzdnu_batched(z_jax, nu_jax, params)
        H_cost_z += w * np.asarray(d2z).reshape(N, n_z, n_z)
        H_cost_nu += w * np.asarray(d2nu).reshape(N, n_nu, n_nu)
        H_cost_znu += w * np.asarray(d2znu).reshape(N, n_z, n_nu)

    for cost_fn in terminal_costs:
        nodes = np.atleast_1d(cost_fn.nodes)
        z_nodes = z_jax[nodes]
        nu_nodes = nu_jax[nodes]
        d2z = cost_fn.d2fcn_dz2_batched(z_nodes, nu_nodes, params)
        d2nu = cost_fn.d2fcn_dnu2_batched(z_nodes, nu_nodes, params)
        d2znu = cost_fn.d2fcn_dzdnu_batched(z_nodes, nu_nodes, params)
        for i, k in enumerate(nodes):
            H_cost_z[k] += np.asarray(d2z[i]).reshape(n_z, n_z)
            H_cost_nu[k] += np.asarray(d2nu[i]).reshape(n_nu, n_nu)
            H_cost_znu[k] += np.asarray(d2znu[i]).reshape(n_z, n_nu)

    return H_cost_z, H_cost_nu, H_cost_znu



def compile_jax_discretization(problem: Problem, method: "SCvx") -> None:
    """Build and cache the JIT-compiled discretization propagator on method."""
    n_z  = problem.index_map.n.z
    n_nu = problem.index_map.n.nu

    # define static indices for stacked RHS vector
    Ak_ind0   = n_z
    Bk_ind0   = Ak_ind0  + n_z*n_z
    Bkp_ind0  = Bk_ind0  + n_z*n_nu

    # pull ltv dynamics
    lin_dyn = problem.constraints.get(type="dynamics")[0].lin_dyn

    # packs the derivative of stacked RHS vector for node k
    def pack_lds_dot(tau, k, lds_k, nu_k, nu_kp, params):

        z       = lds_k[         : Ak_ind0]
        phi_a   = lds_k[Ak_ind0  : Bk_ind0].reshape((n_z, n_z))
        phi_b_m = lds_k[Bk_ind0  : Bkp_ind0].reshape((n_z, n_nu))
        phi_b_p = lds_k[Bkp_ind0 : ].reshape((n_z, n_nu))

        tau_k = k / (problem.index_map.N.time_grid - 1)
        tau_kp = (k+1) / (problem.index_map.N.time_grid - 1)
        a = (tau_kp - tau) / (tau_kp - tau_k)
        b = (tau - tau_k) / (tau_kp - tau_k)
        nu = a * nu_k + b * nu_kp

        fc, Ac, Bc = lin_dyn(z, nu, params)

        P1_dot = fc
        P2_dot = (Ac @ phi_a).reshape((n_z*n_z,))
        P3_dot = (Ac @ phi_b_m + Bc * a).reshape((n_z*n_nu,))
        P4_dot = (Ac @ phi_b_p + Bc * b).reshape((n_z*n_nu,))

        return jnp.concatenate([P1_dot, P2_dot, P3_dot, P4_dot])

    # define dynamics function for diffrax integrator
    def f_dot(tau, lds_k, args):
        k, nu_k, nu_kp, params = args
        return pack_lds_dot(tau, k, lds_k, nu_k, nu_kp, params)

    # initilize stacked propagation vector
    def pack_lds0(z_k):
        P1 = z_k
        P2 = jnp.eye(n_z).reshape(n_z*n_z)
        P3 = jnp.zeros(n_z*n_nu)
        P4 = jnp.zeros(n_z*n_nu)

        return jnp.concatenate([P1, P2, P3, P4])

    # unpacks stacked propagation vector to correct shapes
    def unpack_ldsf(ldsf_k):
        z_minus_k = ldsf_k[ : Ak_ind0]
        A_jax_k    = ldsf_k[Ak_ind0  : Bk_ind0].reshape((n_z, n_z))
        B_jax_k    = ldsf_k[Bk_ind0  : Bkp_ind0].reshape((n_z, n_nu))
        Bp_jax_k   = ldsf_k[Bkp_ind0 : ].reshape((n_z, n_nu))

        return (A_jax_k, B_jax_k, Bp_jax_k, z_minus_k)

    use_fixed_dt = int(getattr(method.flags, "ode_fixed_dt", 0))
    N_grid = problem.index_map.N.time_grid

    if use_fixed_dt:
        nsub = 50
        delta_tau = 1.0 / (N_grid - 1)
        dt_rk4 = delta_tau / nsub

        def propagate_k(k, z_ref, nu_ref, params):
            z_k   = z_ref[k]
            nu_k  = nu_ref[k]
            nu_kp = nu_ref[k+1]
            lds0_k = pack_lds0(z_k)

            tau_k = k / (N_grid - 1)
            taus  = tau_k + jnp.arange(nsub) * dt_rk4

            def rk4_step(lds, tau):
                k1 = pack_lds_dot(tau,              k, lds,                     nu_k, nu_kp, params)
                k2 = pack_lds_dot(tau + dt_rk4/2,   k, lds + (dt_rk4/2) * k1,  nu_k, nu_kp, params)
                k3 = pack_lds_dot(tau + dt_rk4/2,   k, lds + (dt_rk4/2) * k2,  nu_k, nu_kp, params)
                k4 = pack_lds_dot(tau + dt_rk4,     k, lds + dt_rk4 * k3,      nu_k, nu_kp, params)
                return lds + (dt_rk4 / 6) * (k1 + 2*k2 + 2*k3 + k4), None

            ldsf_k, _ = jax.lax.scan(rk4_step, lds0_k, taus)
            return unpack_ldsf(ldsf_k)

    else:
        def propagate_k(k, z_ref, nu_ref, params):
            z_k   = z_ref[k]
            nu_k  = nu_ref[k]
            nu_kp = nu_ref[k+1]
            lds0_k = pack_lds0(z_k)

            tau_k  = k / (N_grid - 1)
            tau_kp = (k+1) / (N_grid - 1)

            term = diffrax.ODETerm(f_dot)
            solver = diffrax.Dopri5()
            stepsize_controller = diffrax.PIDController(rtol=1e-7, atol=1e-7)
            sol = diffrax.diffeqsolve(
                term,
                solver,
                tau_k,
                tau_kp,
                0.00005,
                lds0_k,
                stepsize_controller=stepsize_controller,
                args=(k, nu_k, nu_kp, params),
                max_steps=65536,
            )
            ldsf_k = sol.ys[-1]
            return unpack_ldsf(ldsf_k)

    propagate = jax.jit(jax.vmap(propagate_k, in_axes=(0, None, None, None)))
    method.propagate_discretization_jax = propagate

def compile_rk4_discretization(problem, method):

    dyn_fcn = jax.jit(problem.constraints.get(type='dynamics')[0].fcn)
    N_grid  = problem.index_map.N.time_grid
    nsub    = 2
    delta_tau = 1.0 / (N_grid - 1)
    dt_rk4  = delta_tau / nsub

    def f_dot(k, tau, z, nu_k, nu_kp, params):
        tau_k  = k / (N_grid - 1)
        tau_kp = (k+1) / (N_grid - 1)
        a      = (tau_kp - tau) / (tau_kp - tau_k)
        b      = (tau - tau_k)  / (tau_kp - tau_k)
        nu     = a * nu_k + b * nu_kp
        fc = dyn_fcn(z, nu, params)
        return fc

    def rk4_step(carry, tau):
        z, k, nu_k, nu_kp, params = carry
        k1 = f_dot(k, tau,            z,                    nu_k, nu_kp, params)
        k2 = f_dot(k, tau + dt_rk4/2, z + (dt_rk4/2) * k1, nu_k, nu_kp, params)
        k3 = f_dot(k, tau + dt_rk4/2, z + (dt_rk4/2) * k2, nu_k, nu_kp, params)
        k4 = f_dot(k, tau + dt_rk4,   z +     dt_rk4 * k3, nu_k, nu_kp, params)
        z_next = z + (dt_rk4 / 6) * (k1 + 2*k2 + 2*k3 + k4)
        return (z_next, k, nu_k, nu_kp, params), None

    def propagate_k(k, z_k, nu_k, nu_kp, params):
        tau_k = k / (N_grid - 1)
        taus  = tau_k + jnp.arange(nsub) * dt_rk4
        carry_init = (z_k, k, nu_k, nu_kp, params)
        (z_kp, _, _, _, _), _ = jax.lax.scan(rk4_step, carry_init, taus)
        return z_kp

    def lagrangian_k(k, lam_k, z_k, nu_k, nu_kp, params):
        return -lam_k @ propagate_k(k, z_k, nu_k, nu_kp, params)

    prop_jacobians_k = jax.jacfwd(propagate_k, argnums=(1, 2, 3))
    cnstr_hessians_k = jax.hessian(lagrangian_k, argnums=(2, 3, 4))

    method.propagate           = jax.jit(jax.vmap(propagate_k,      in_axes=(0, 0, 0, 0, None)))
    method.propagate_jacobians = jax.jit(jax.vmap(prop_jacobians_k, in_axes=(0, 0, 0, 0, None)))
    method.cnstr_hessians      = jax.jit(jax.vmap(cnstr_hessians_k, in_axes=(0, 0, 0, 0, 0, None)))

# inverse free discretize with jax
def discretize_ms_variational(z_ref_np, nu_ref_np, problem, method):
    """
    Returns:
        Tuple of (Ak, Bk, Bkp, z_minus).

    """
    # convert numpy arrays to jax
    z_ref = jnp.asarray(z_ref_np)
    nu_ref = jnp.asarray(nu_ref_np)
    params = problem.params

    # TODO(Skye): ensure dtk is computed from the z vector correctly for each node in the jax propagator
    # call jitted propagator for each node
    ks = jnp.arange(method.index_map.N.time_grid - 1)
    A_jax, B_jax, Bp_jax, z_minus = method.propagate_discretization_jax(ks, z_ref, nu_ref, params)

    z_ref_0 = z_ref[[0], :]

    return np.asarray(A_jax), np.asarray(B_jax), np.asarray(Bp_jax), np.asarray(jnp.vstack([z_ref_0, z_minus]))

# inverse free discretize with jax
def discretize_ms_rk4(z_ref_np, nu_ref_np, problem, method):

    # convert numpy arrays to jax
    z_ref_ks   = jnp.asarray(z_ref_np[:-1])
    z_ref_kps  = jnp.asarray(z_ref_np[1:])
    nu_ref_ks  = jnp.asarray(nu_ref_np[:-1])
    nu_ref_kps = jnp.asarray(nu_ref_np[1:])
    lam_refs   = jnp.asarray(method.lagrangian_duals.dynamics)
    params     = problem.params

    ks = jnp.arange(method.index_map.N.time_grid - 1)

    z_minus              = method.propagate(ks, z_ref_ks, nu_ref_ks, nu_ref_kps, params)
    A_jax, B_jax, Bp_jax = method.propagate_jacobians(ks, z_ref_ks, nu_ref_ks, nu_ref_kps, params)

    method.H_z_k, method.H_nu_k, method.H_nu_kp = method.cnstr_hessians(ks, lam_refs, z_ref_ks, nu_ref_ks, nu_ref_kps, params)

    z_ref_0 = z_ref_ks[[0], :]
    
    return np.asarray(A_jax), np.asarray(B_jax), np.asarray(Bp_jax), np.asarray(jnp.vstack([z_ref_0, z_minus]))


def compute_linsys_discrete(
    z_ref: np.ndarray, nu_ref: np.ndarray, problem: Problem, method: "SCvx",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the linear system in discrete form.

    Parameters
    ----------
        z_ref (numpy.ndarray): Reference state trajectory.
        nu_ref (numpy.ndarray): Reference control trajectory.
        t_ref (numpy.ndarray): Node time values.
        trajopt_obj (dict): Dictionary containing trajopt_obj parameters.

    Returns
    -------
        tuple: Ak, Bk, Bkp, z_minus.

    """

    Ak, Bk, Bkp, z_minus = discretize_ms_rk4(z_ref, nu_ref, problem, method)

    return Ak, Bk, Bkp, z_minus

def build_ps_dyn_constraints(subproblem: "SCvx") -> cp.Constraint:
    """Build pseudospectral dynamics constraints as a single block matrix equation.

    At collocation node k: state is z[k+1], control is nu[k+1] (both at etau[k+1]).

    Returns:
        cp.Constraint: Single constraint for all collocation nodes, shape (N_col, n_z).

    """
    N_col  = subproblem.index_map.N.time_grid - 1

    Z   = subproblem.cp_params.z_ref + subproblem.dz
    lhs = 2.0 * (subproblem.ps_D @ Z)

    rhs_list = []
    for k in range(N_col):
        rhs_k = subproblem.cp_params.ps_f_ref[k] + subproblem.cp_params.ps_Ac[k] @ subproblem.dz[k + 1] + subproblem.cp_params.ps_Bc[k] @ subproblem.dnu[k + 1]
        rhs_list.append(rhs_k)

    rhs = cp.vstack(rhs_list)

    return lhs == rhs


def compute_ps_dynamics_and_jacobians(
    z_ref: np.ndarray, nu_ref: np.ndarray, problem: Problem, method: "SCvx",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute dynamics and Jacobians at reference trajectory for pseudospectral collocation.

    State and control at collocation node k are both at etau[k+1]: z[k+1], nu[k+1].

    Returns:
        Tuple of (f_ref_col, Ac_col, Bc_col).

    """
    N_col = problem.index_map.N.time_grid - 1
    n_z = problem.index_map.n.z
    n_nu = problem.index_map.n.nu

    lin_dyn = problem.constraints.get(type="dynamics")[0].lin_dyn
    params = problem.params

    f_ref_col = np.zeros((N_col, n_z))
    Ac_col = np.zeros((N_col, n_z, n_z))
    Bc_col = np.zeros((N_col, n_z, n_nu))

    for k in range(N_col):
        z_k = np.asarray(z_ref[k + 1])
        nu_k = np.asarray(nu_ref[k + 1])

        fc_k, Ac_k, Bc_k = lin_dyn(z_k, nu_k, params)

        f_ref_col[k, :] = np.asarray(fc_k)
        Ac_col[k, :, :] = np.asarray(Ac_k)
        Bc_col[k, :, :] = np.asarray(Bc_k)

    return f_ref_col, Ac_col, Bc_col


def compute_ps_differentiation_matrix(
    N_col: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the pseudospectral differentiation matrix D for fLGR collocation.

    Returns:
        Tuple of (tau, etau, w, D).

    """
    tau, etau, w, D = pseudospectral.flipped_radau_differential_operator(N_col)
    return tau, etau, w, D



##############################################################################################################
# ORIGINAL DISCRETIZATION METHODS - DEPRECATED
##############################################################################################################

# TODO(Skye): Clean up and make work for numpy or sympy! Currently deprecated


# def _coerce_time_grid(t_ref, N):
#     t_arr = np.asarray(t_ref).reshape(-1)

#     # Transition helper:
#     # - preferred input is node times of length N
#     # - legacy support for interval lengths of length N-1
#     if t_arr.size == N - 1:
#         return np.concatenate(([0.0], np.cumsum(t_arr)))

#     if t_arr.size != N:
#         raise ValueError(f"Expected t_ref with length {N} (or legacy dt_ref with length {N-1}), got {t_arr.size}.")

#     return t_arr

# def compute_linsys_discrete(z_ref, nu_ref, t_ref, problem, method):
#     """
#     Compute the linear system in discrete form.

#     Parameters:
#     z_ref (numpy.ndarray): Reference state trajectory.
#     nu_ref (numpy.ndarray): Reference control trajectory.
#     t_ref (numpy.ndarray): Node time values.
#     trajopt_obj (dict): Dictionary containing trajopt_obj parameters.

#     Returns:
#     tuple: Ak, Bk, Bkp, z_minus
#     """

#     if method.flags.jax_dyn:
#         Ak, Bk, Bkp, z_minus = discretize_inv_free_jax(z_ref, nu_ref, problem, method)
#     else:
#         t_ref = _coerce_time_grid(t_ref, method.index_map.N.time_grid)
#         if method.flags.ctcs != "none":
#             Ak, Bk, Bkp, z_minus = discretize_ctcs(z_ref, nu_ref, t_ref, problem, method)
#         else:
#             Ak, Bk, Bkp, z_minus = discretize_inv_foh(z_ref, nu_ref, t_ref, problem, method)

#         return Ak, Bk, Bkp, z_minus

#     return Ak, Bk, Bkp, z_minus

# Compute exact discretize for linear dynamic system
# def discretize_inv_foh(z_ref, nu_ref, t_ref, problem, method):
#     N = method.index_map.N.time_grid

#     traj_minus_data = {"z_minus": [z_ref[0]]}

#     # Precompute
#     eye_flat = np.eye(problem.index_map.n.state).ravel()

#     # Stack dynamics
#     lds0_stack = []
#     for k in range(N - 1):
#         method.lds0[method.z_ind] = z_ref[k]
#         method.lds0[method.Ak_ind] = eye_flat
#         lds0_stack.append(method.lds0.copy())

#     lds0_stack = np.concatenate(lds0_stack)

#     def derivs_step(tau, lds):
#         return RHS_ltv(tau, lds, nu_ref, t_ref, problem, method)

#     sol = solve_ivp(derivs_step, [0, 1], lds0_stack, method="RK45", atol=1e-6, rtol=1e-6)

#     lds_end = sol.y[:, -1]  # shape: (total_state_size,)

#     assert lds_end.shape[0] == (N - 1) * method.lds0_size

#     Ak  = np.zeros((N - 1, problem.index_map.n.state, problem.index_map.n.state))
#     Bk  = np.zeros((N - 1, problem.index_map.n.state, problem.index_map.n.nu))
#     Bkp = np.zeros((N - 1, problem.index_map.n.state, problem.index_map.n.nu))

#     for k in range(N - 1):
#         base    = k * method.lds0_size
#         traj_minus_data["z_minus"].append(lds_end[base + method.z_ind])

#         Ak_bar  = lds_end[base + method.Ak_ind].reshape(problem.index_map.n.state, problem.index_map.n.state)
#         Bk_bar  = lds_end[base + method.Bk_ind].reshape(problem.index_map.n.state, problem.index_map.n.nu)
#         Bkp_bar = lds_end[base + method.Bkp_ind].reshape(problem.index_map.n.state, problem.index_map.n.nu)

#         Ak[k]   = Ak_bar
#         Bk[k]   = Ak_bar @ Bk_bar
#         Bkp[k]  = Ak_bar @ Bkp_bar

#     z_minus    = np.array(traj_minus_data["z_minus"])

#     return Ak, Bk, Bkp, z_minus

# # Integrate linear system
# def RHS_ltv(tau, lds, nu_ref, t_ref, problem, method):



#     # Initialize
#     lds_dot         = np.zeros_like(lds)
#     N               = method.index_map.N.time_grid

#     # Extract times and FOH control input
#     Om_k            = 1 - tau
#     Om_kp           = tau

#     nrows, ncols    = nu_ref.shape
#     rows            = nrows if nrows > ncols else ncols

#     v_1             = Om_k * np.ones((rows, 1))
#     v_2             = Om_kp * np.ones((rows - 1, 1))
#     Om              = np.diagflat(v_1) + np.diagflat(v_2, 1)

#     nu              = Om @ nu_ref

#     for k in range(N - 1):
#         dt_k       = t_ref[k + 1] - t_ref[k]

#         # Extract state info
#         z_k         = lds[ k * method.lds0_size + method.z_ind ]

#         # Extract continuous time Jacobians
#         fc, Ac, Bc = problem.lin_dyn(tau, z_k, nu[k])

#         # Extract STM
#         Phi_tau     = lds[ k * method.lds0_size + method.Ak_ind ].reshape(problem.index_map.n.state, problem.index_map.n.state)

#         # Construct Jacobians w.r.t. tau
#         f_tau       = dt_k * fc
#         A_tau       = dt_k * Ac
#         B_tau       = dt_k * Om_k * Bc
#         Bp_tau      = dt_k * Om_kp * Bc

#         Phi_tau_inv = np.linalg.pinv(Phi_tau)

#         # Construct derivatives
#         z_dot       = f_tau
#         A_tau_dot   = A_tau @ Phi_tau
#         B_tau_dot   = Phi_tau_inv @ B_tau
#         Bp_tau_dot  = Phi_tau_inv @ Bp_tau

#         # Setup linear system properly
#         lds_dot[ k * method.lds0_size + method.z_ind  ] = z_dot
#         lds_dot[ k * method.lds0_size + method.Ak_ind ] = A_tau_dot.flatten()
#         lds_dot[ k * method.lds0_size + method.Bk_ind ] = B_tau_dot.flatten()
#         lds_dot[ k * method.lds0_size + method.Bkp_ind] = Bp_tau_dot.flatten()

#     return lds_dot


