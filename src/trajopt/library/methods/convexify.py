import numpy as np
import sympy as sp
import jax
import jax.numpy as jnp
import cvxpy as cp

# jax autodiff for affine approximations
def linearize_jax(fcn):

    dfcn_dz = jax.jit(jax.jacrev(fcn, argnums=1))
    dfcn_du = jax.jit(jax.jacrev(fcn, argnums=2))
    f = jax.jit(fcn)

    return f, dfcn_dz, dfcn_du

def linearize_jax_ctcs(fcn, constraints, n):

    def wrapped_fcn(t, z, nu, params):
        constr = jnp.concatenate([constraint.fcn(t, z[:n], nu, params) for constraint in constraints.get("ct")])
        f_val = jnp.concatenate([fcn(t, z[:n], nu, params), jnp.square(jnp.maximum(constr, 0.0))])
        return f_val

    dfcn_dz = jax.jacrev(wrapped_fcn, argnums=1)
    dfcn_du = jax.jacrev(wrapped_fcn, argnums=2)
    f = jax.jit(wrapped_fcn)
    
    return f, dfcn_dz, dfcn_du

# PROTOTYPE 
def linearize_sympy(fcn, trajopt_obj):
    z, nu = trajopt_obj.method.z_init, trajopt_obj.method.nu_init
    n = trajopt_obj.model.n
    m = trajopt_obj.model.m

    z_sym = sp.symbols(f"z0:{n}")
    nu_sym = sp.symbols(f"u0:{m}")
    fcn_sym = fcn(0, z_sym, nu_sym, trajopt_obj)
    dfcn_dz_sym = sp.diff(fcn_sym, z_sym)
    dfcn_du_sym = sp.diff(fcn_sym, nu_sym)

    return fcn_sym, dfcn_dz_sym, dfcn_du_sym

def compute_ctcs_jacobians(t, z, nu, trajopt_obj):
    mission = trajopt_obj.mission
    model = trajopt_obj.model
    method = trajopt_obj.method

    n = model.n
    n_ineq = mission.n_ineq

    # Evaluate linearized dynamics
    f_xu, dfdx, dfdnu = model.lin_dyn(t, z[:n], nu)

    # Evaluate all nonconvex inequality constraints and stack them
    g_xu = np.zeros(n_ineq)
    dgdx = np.zeros((n_ineq, n))
    dgdnu = np.zeros((n_ineq, model.m))


    # TODO (CARLOS): use Skye's indexing to make this cleaner? can we get rid of this loop?
    col_start = 0
    for constraint in model.nonconvex_inequality_constraints:
        col_end = col_start + constraint.dimension
        
        f, dfcn_dz, dfcn_du = constraint.affine_approximation(t, z, nu)
        g_xu[col_start:col_end] = np.asarray(f).flatten()
        dgdx[col_start:col_end, :] = np.asarray(dfcn_dz)
        dgdnu[col_start:col_end, :] = np.asarray(dfcn_du)
        
        col_start = col_end
    
    # Apply CTCS weight
    g_xu = g_xu * method.weights["w_ctcs"]
    dgdx = dgdx * method.weights["w_ctcs"]

    # Conditional constraint smoothing
    beta_dot = np.maximum(g_xu, 0.0) ** 2
    zeta = np.concatenate([f_xu, beta_dot])

    # Active constraint mask
    mask = g_xu > 0
    D = np.diag(mask.astype(float))

    # Constraint Jacobians
    dbetadot_dx = 2.0 * D @ np.diag(g_xu) @ dgdx
    dbetadot_du = 2.0 * D @ np.diag(g_xu) @ dgdnu

    # Stack into full Jacobians
    dzeta_dz = np.block([
        [dfdx,               np.zeros((n, n_ineq))],
        [dbetadot_dx,        np.zeros((n_ineq, n_ineq))]
    ])

    dzeta_du = np.vstack([
        dfdnu,
        dbetadot_du
    ])

    Ac = dzeta_dz
    Bc = dzeta_du
    fc = zeta

    return Ac, Bc, fc