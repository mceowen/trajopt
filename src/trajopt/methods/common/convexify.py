import jax
import jax.numpy as jnp
import numpy as np
import cvxpy as cp


# Taylor-expand g(z, nu) once per iteration: g ~= g0 + dg/dz @ dz + dg/dnu @ dnu.
# The convex subproblem constrains that affine form, not g itself.

def compile_affine(constraint, subproblem):
    """JIT + batch the fcn and its z/nu Jacobians for first-order Taylor linearization."""
    fcn = constraint.constraint.fcn_znu
    g      = jax.jit(fcn)
    dg_dz  = jax.jit(jax.jacfwd(fcn, argnums=0))
    dg_dnu = jax.jit(jax.jacfwd(fcn, argnums=1))
    g_batched      = jax.jit(jax.vmap(g,      in_axes=(0, 0, None)))
    dg_dz_batched  = jax.jit(jax.vmap(dg_dz,  in_axes=(0, 0, None)))
    dg_dnu_batched = jax.jit(jax.vmap(dg_dnu, in_axes=(0, 0, None)))

    constraint.fcn_batched = g_batched

    def g_aff_batched(z, nu, params, g=g_batched, dz=dg_dz_batched, dnu=dg_dnu_batched):
        return g(z, nu, params), dz(z, nu, params), dnu(z, nu, params)
    constraint.g_aff_batched = g_aff_batched


def compile_affine_second_order(constraint, subproblem):
    """Extends compile_affine with the per-node Lagrangian Hessian a second-order trust region needs."""
    compile_affine(constraint, subproblem)

    fcn = constraint.constraint.fcn_znu

    def lagrangian_k(lam, z, nu, params, fcn=fcn):
        return lam @ fcn(z, nu, params)
    constraint.lagrangian_hessians = jax.jit(jax.vmap(
        jax.hessian(lagrangian_k, argnums=(1, 2)), in_axes=(0, 0, 0, None),
    ))


def create_cvxpy_parameters_affine(constraint, subproblem):
    if constraint.shape is None:
        return
    n_z  = subproblem.index_map.n.z
    n_nu = subproblem.index_map.n.nu
    dim  = constraint.constraint.dimension
    nn   = len(constraint.nodes)
    constraint.dgdz_param  = cp.Parameter((nn, dim, n_z),  name=f"dgdz_{constraint.name}")
    constraint.dgdnu_param = cp.Parameter((nn, dim, n_nu), name=f"dgdnu_{constraint.name}")
    constraint.g0_param    = cp.Parameter((nn, dim),       name=f"g0_{constraint.name}")


def create_cvxpy_constraints_affine_inequality(constraint, subproblem):
    if constraint.shape is None:
        return
    vb = constraint.penalties.vb_var
    constraint.cp_ineq_constraints = []
    for i, k in enumerate(constraint.nodes):
        g_lin = (
            constraint.dgdz_param[i] @ subproblem.dz[k, :]
            + constraint.dgdnu_param[i] @ subproblem.dnu[k, :]
            + constraint.g0_param[i]
        )
        cnst = (g_lin - vb[i] <= 0) if vb is not None else (g_lin <= 0)
        constraint.cp_ineq_constraints.append(cnst)
        subproblem.cp_constraints.append(cnst)
        if vb is not None:
            subproblem.cp_constraints.append(vb[i] >= 0)


def create_cvxpy_constraints_affine_equality(constraint, subproblem):
    if constraint.shape is None:
        return
    vb = constraint.penalties.vb_var
    constraint.cp_eq_constraints = []
    for i, k in enumerate(constraint.nodes):
        g_lin = (
            constraint.dgdz_param[i] @ subproblem.dz[k, :]
            + constraint.dgdnu_param[i] @ subproblem.dnu[k, :]
            + constraint.g0_param[i]
        )
        cnst = (g_lin - vb[i] == 0) if vb is not None else (g_lin == 0)
        constraint.cp_eq_constraints.append(cnst)
        subproblem.cp_constraints.append(cnst)


def update_cvxpy_parameters_affine(constraint, subproblem):
    if not hasattr(constraint, 'g0_param'):
        return
    z      = jnp.asarray(subproblem.current_iter_data.z_opt)
    nu     = jnp.asarray(subproblem.current_iter_data.nu_opt)
    params = subproblem.params
    g, dgdz, dgdnu = constraint.g_aff_batched(z[constraint.nodes], nu[constraint.nodes], params)
    constraint.g0_param.value    = np.asarray(g)
    constraint.dgdz_param.value  = np.asarray(dgdz)
    constraint.dgdnu_param.value = np.asarray(dgdnu)


def _update_current_iter_data_affine(constraint, subproblem):
    z      = jnp.asarray(subproblem.current_iter_data.z_opt)
    nu     = jnp.asarray(subproblem.current_iter_data.nu_opt)
    params = subproblem.params
    constraint.g_nl = np.asarray(constraint.fcn_batched(z[constraint.nodes], nu[constraint.nodes], params))


def update_current_iter_data_affine_inequality(constraint, subproblem):
    if constraint.cp_ineq_constraints is None:
        return
    _update_current_iter_data_affine(constraint, subproblem)


def update_current_iter_data_affine_equality(constraint, subproblem):
    if constraint.cp_eq_constraints is None:
        return
    _update_current_iter_data_affine(constraint, subproblem)


def _update_current_iter_data_affine_second_order(constraint, subproblem, cnstr_list):
    """EMA-updates the per-node Lagrange multiplier used by the Hessian."""
    if not cnstr_list:
        return
    alpha = subproblem.current_iter_data.get("alpha", 1.0)
    lam = np.array([c.dual_value for c in cnstr_list])
    constraint.lagrangian_dual = (1.0 - alpha) * constraint.lagrangian_dual + alpha * lam


def update_current_iter_data_affine_inequality_second_order(constraint, subproblem):
    update_current_iter_data_affine_inequality(constraint, subproblem)
    _update_current_iter_data_affine_second_order(constraint, subproblem, constraint.cp_ineq_constraints)


def update_current_iter_data_affine_equality_second_order(constraint, subproblem):
    update_current_iter_data_affine_equality(constraint, subproblem)
    _update_current_iter_data_affine_second_order(constraint, subproblem, constraint.cp_eq_constraints)
