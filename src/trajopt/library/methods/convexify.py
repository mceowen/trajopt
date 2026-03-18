import numpy as np
import sympy as sp
import jax
import jax.numpy as jnp
import cvxpy as cp

# jax autodiff for affine approximations
def linearize_jax(fcn):

    dfcn_dz     = jax.jit(jax.jacrev(fcn, argnums=1))
    dfcn_dnu    = jax.jit(jax.jacrev(fcn, argnums=2))
    f           = jax.jit(fcn)

    return f, dfcn_dz, dfcn_dnu

# def linearize_jax_ctcs(fcn, constraints):

#     ctcs_constraints = tuple(constraints.get(ct=1))

#     constr_fcn = constraints.index_map.wrap_txu_fcn(
#         lambda t, x, u, params: jnp.concatenate(
#             [constraint.fcn(t, x, u, params) for constraint in ctcs_constraints]
#         )
#     )

#     # TODO(Skye): Verify jax ctcs integration below
#     def augmented_fcn(t, z, nu, params):
#         z_ctcs_idx = constraints.index_map.indices.z.ctcs
#         f_val = fcn(z, nu, params)

#         # TODO(Skye): remove and properly implement time dilation
#         # Potentially add analytical jacobians for CTCS portion because we have those,
#         # but use jax for constraint derivs
#         if len(z_ctcs_idx) > 0:
#             f_val = f_val.at[z_ctcs_idx].set(jnp.maximum(100 * constr_fcn(z, nu, params), 0.0))

#         return f_val

#     dfcn_dz     = jax.jit(jax.jacrev(augmented_fcn, argnums=1))
#     dfcn_dnu    = jax.jit(jax.jacrev(augmented_fcn, argnums=2))
#     f           = jax.jit(augmented_fcn)
    
#     return f, dfcn_dz, dfcn_dnu

# PROTOTYPE 
def linearize_sympy(fcn, trajopt_obj):
    z, nu = trajopt_obj.method.initial_guess.z, trajopt_obj.method.initial_guess.nu
    n = trajopt_obj.model.n
    m = trajopt_obj.model.m

    z_sym = sp.symbols(f"z0:{n}")
    nu_sym = sp.symbols(f"u0:{m}")
    fcn_sym = fcn(0, z_sym, nu_sym, trajopt_obj)
    dfcn_dz_sym = sp.diff(fcn_sym, z_sym)
    dfcn_dnu_sym = sp.diff(fcn_sym, nu_sym)

    return fcn_sym, dfcn_dz_sym, dfcn_dnu_sym


# TODO(Skye): Add back in analytical CTCS Jacobians. Try to use as unit test for jax.