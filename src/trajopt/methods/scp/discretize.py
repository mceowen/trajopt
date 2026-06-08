import types
from typing import TYPE_CHECKING

import cvxpy as cp
import diffrax
import jax
import jax.numpy as jnp
import numpy as np

from trajopt.segment import Segment
from trajopt.methods.scp import pseudospectral

if TYPE_CHECKING:
    from trajopt.methods.scp.subproblem import Subproblem

jax.config.update("jax_enable_x64", True)


def compile_constraint_linearizations(segment: Segment, subprob: "Subproblem") -> None:
    """Compile JAX linearizations of the dynamics and nonconvex inequality constraints (once).

    The constraint objects only carry their plain functions; all jitted Jacobians, batched
    evaluators, and Lagrangian Hessians used by the SCP iterations are built here and cached
    on the subprob.
    """
    dynamics = next(c for c in segment.constraints.values() if c.type == "dynamics")
    f = jax.jit(dynamics.fcn)
    df_dz = jax.jit(jax.jacfwd(dynamics.fcn, argnums=0))
    df_dnu = jax.jit(jax.jacfwd(dynamics.fcn, argnums=1))
    subprob.lin_dyn = lambda z, nu, params: (f(z, nu, params), df_dz(z, nu, params), df_dnu(z, nu, params))

    subprob.nonconvex_lin = {}
    for c in [c for c in segment.constraints.values() if c.type == "nonconvex_inequality" and getattr(c, "ct", 0) == 0]:
        g = jax.jit(c.fcn_znu)
        dg_dz = jax.jit(jax.jacfwd(c.fcn_znu, argnums=0))
        dg_dnu = jax.jit(jax.jacfwd(c.fcn_znu, argnums=1))
        g_batched     = jax.jit(jax.vmap(g,      in_axes=(0, 0, None)))
        dg_dz_batched = jax.jit(jax.vmap(dg_dz,  in_axes=(0, 0, None)))
        dg_dnu_batched = jax.jit(jax.vmap(dg_dnu, in_axes=(0, 0, None)))

        fcn = c.fcn_znu
        def lagrangian_k(lam, z, nu, params, fcn=fcn):
            return lam @ fcn(z, nu, params)
        lagrangian_hessians = jax.jit(jax.vmap(jax.hessian(lagrangian_k, argnums=(1, 2)), in_axes=(0, 0, 0, None)))

        subprob.nonconvex_lin[c.name] = types.SimpleNamespace(
            g_aff_batched=lambda z, nu, params, g=g_batched, dz=dg_dz_batched, dnu=dg_dnu_batched: (
                g(z, nu, params), dz(z, nu, params), dnu(z, nu, params)
            ),
            fcn_batched=g_batched,
            lagrangian_hessians=lagrangian_hessians,
        )

def compute_nonconvex_costs(z, nu, segment, subprob):
    N         = segment.index_map.N.all
    n_z       = segment.index_map.n.z
    n_nu       = segment.index_map.n.nu

    cost     = np.zeros((N, 1))
    dcostdz  = np.zeros((N, 1, n_z))
    dcostdnu = np.zeros((N, 1, n_nu))

    params = segment.params
    nonconvex_costs = [c for c in segment.costs.values() if c.type == "nonconvex"]
    terminal_costs  = [c for c in segment.costs.values() if c.type == "terminal"]
    running_costs   = [c for c in segment.costs.values() if c.type == "running"]

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


def compute_nonconvex_cost_hessians(z, nu, segment, subprob):
    N = segment.index_map.N.all
    n_z = segment.index_map.n.z
    n_nu = segment.index_map.n.nu

    H_cost_z = np.zeros((N, n_z, n_z))
    H_cost_nu = np.zeros((N, n_nu, n_nu))
    H_cost_znu = np.zeros((N, n_z, n_nu))

    params = segment.params
    nonconvex_costs = [c for c in segment.costs.values() if c.type == "nonconvex"]
    terminal_costs = [c for c in segment.costs.values() if c.type == "terminal"]
    running_costs = [c for c in segment.costs.values() if c.type == "running"]

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



def compile_jax_discretization(segment: Segment, subprob: "Subproblem") -> None:
    """Build and cache the JIT-compiled discretization propagator on subprob."""
    n_z  = segment.index_map.n.z
    n_nu = segment.index_map.n.nu

    # define static indices for stacked RHS vector
    Ak_ind0   = n_z
    Bk_ind0   = Ak_ind0  + n_z*n_z
    Bkp_ind0  = Bk_ind0  + n_z*n_nu

    # pull ltv dynamics
    lin_dyn = subprob.lin_dyn

    # packs the derivative of stacked RHS vector for node k
    def pack_lds_dot(tau, k, lds_k, nu_k, nu_kp, params):

        z       = lds_k[         : Ak_ind0]
        phi_a   = lds_k[Ak_ind0  : Bk_ind0].reshape((n_z, n_z))
        phi_b_m = lds_k[Bk_ind0  : Bkp_ind0].reshape((n_z, n_nu))
        phi_b_p = lds_k[Bkp_ind0 : ].reshape((n_z, n_nu))

        tau_k = k / (segment.index_map.N.all - 1)
        tau_kp = (k+1) / (segment.index_map.N.all - 1)
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

    use_fixed_dt = True
    N_grid = segment.index_map.N.all

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
    subprob.propagate_discretization_jax = propagate

def compile_rk4_discretization(segment, subprob):

    dyn_fcn = jax.jit(next(c for c in segment.constraints.values() if c.type == "dynamics").fcn)
    N_grid  = segment.index_map.N.all
    nsub    = 10
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

    subprob.propagate           = jax.jit(jax.vmap(propagate_k,      in_axes=(0, 0, 0, 0, None)))
    subprob.propagate_jacobians = jax.jit(jax.vmap(prop_jacobians_k, in_axes=(0, 0, 0, 0, None)))
    subprob.cnstr_hessians      = jax.jit(jax.vmap(cnstr_hessians_k, in_axes=(0, 0, 0, 0, 0, None)))

# inverse free discretize with jax
def discretize_ms_variational(z_ref_np, nu_ref_np, segment, subprob):
    """
    Returns:
        Tuple of (Ak, Bk, Bkp, z_minus).

    """
    # convert numpy arrays to jax
    z_ref = jnp.asarray(z_ref_np)
    nu_ref = jnp.asarray(nu_ref_np)
    params = segment.params

    # TODO(Skye): ensure dtk is computed from the z vector correctly for each node in the jax propagator
    # call jitted propagator for each node
    ks = jnp.arange(segment.index_map.N.all - 1)
    A_jax, B_jax, Bp_jax, z_minus = subprob.propagate_discretization_jax(ks, z_ref, nu_ref, params)

    z_ref_0 = z_ref[[0], :]

    return np.asarray(A_jax), np.asarray(B_jax), np.asarray(Bp_jax), np.asarray(jnp.vstack([z_ref_0, z_minus]))

# inverse free discretize with jax
def discretize_ms_rk4(z_ref_np, nu_ref_np, segment, subprob):

    # convert numpy arrays to jax
    z_ref_ks   = jnp.asarray(z_ref_np[:-1])
    z_ref_kps  = jnp.asarray(z_ref_np[1:])
    nu_ref_ks  = jnp.asarray(nu_ref_np[:-1])
    nu_ref_kps = jnp.asarray(nu_ref_np[1:])
    lam_refs   = jnp.asarray(subprob.lagrangian_duals.dynamics)
    params     = segment.params

    ks = jnp.arange(segment.index_map.N.all - 1)

    z_minus              = subprob.propagate(ks, z_ref_ks, nu_ref_ks, nu_ref_kps, params)
    A_jax, B_jax, Bp_jax = subprob.propagate_jacobians(ks, z_ref_ks, nu_ref_ks, nu_ref_kps, params)

    subprob.H_z_k, subprob.H_nu_k, subprob.H_nu_kp = subprob.cnstr_hessians(ks, lam_refs, z_ref_ks, nu_ref_ks, nu_ref_kps, params)

    z_ref_0 = z_ref_ks[[0], :]
    
    return np.asarray(A_jax), np.asarray(B_jax), np.asarray(Bp_jax), np.asarray(jnp.vstack([z_ref_0, z_minus]))


def compute_linsys_discrete(
    z_ref: np.ndarray, nu_ref: np.ndarray, segment: Segment, subprob: "Subproblem",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the linear system in discrete form.

    Parameters
    ----------
        z_ref (numpy.ndarray): Reference state trajectory.
        nu_ref (numpy.ndarray): Reference control trajectory.
        t_ref (numpy.ndarray): Node time values.
        segment: Segment model for discretization.
        subprob: Segment subproblem parameters for discretization.

    Returns
    -------
        tuple: Ak, Bk, Bkp, z_minus.

    """

    Ak, Bk, Bkp, z_minus = discretize_ms_rk4(z_ref, nu_ref, segment, subprob)

    return Ak, Bk, Bkp, z_minus

def build_ps_dyn_constraints(subprob: "Subproblem") -> cp.Constraint:
    """Build pseudospectral dynamics constraints as a single block matrix equation.

    At collocation node k: state is z[k+1], control is nu[k+1] (both at etau[k+1]).

    Returns:
        cp.Constraint: Single constraint for all collocation nodes, shape (N_col, n_z).

    """
    N_col  = subprob.segment.index_map.N.all - 1

    Z   = subprob.cp_params.z_ref + subprob.dz
    lhs = 2.0 * (subprob.ps_D @ Z)

    rhs_list = []
    for k in range(N_col):
        rhs_k = subprob.cp_params.ps_f_ref[k] + subprob.cp_params.ps_Ac[k] @ subprob.dz[k + 1] + subprob.cp_params.ps_Bc[k] @ subprob.dnu[k + 1]
        rhs_list.append(rhs_k)

    rhs = cp.vstack(rhs_list)

    return lhs == rhs


def compute_ps_dynamics_and_jacobians(
    z_ref: np.ndarray, nu_ref: np.ndarray, segment: Segment, subprob: "Subproblem",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute dynamics and Jacobians at reference trajectory for pseudospectral collocation.

    State and control at collocation node k are both at etau[k+1]: z[k+1], nu[k+1].

    Returns:
        Tuple of (f_ref_col, Ac_col, Bc_col).

    """
    N_col = segment.index_map.N.all - 1
    n_z = segment.index_map.n.z
    n_nu = segment.index_map.n.nu

    lin_dyn = subprob.lin_dyn
    params = segment.params

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
