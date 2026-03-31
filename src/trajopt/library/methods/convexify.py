import numpy as np
import sympy as sp
import jax
import jax.numpy as jnp
import cvxpy as cp

# jax autodiff for affine approximations
def linearize_jax(fcn):

    dfcn_dz = jax.jit(jax.jacfwd(fcn, argnums=1))
    dfcn_dnu = jax.jit(jax.jacfwd(fcn, argnums=2))
    f = jax.jit(fcn)

    return f, dfcn_dz, dfcn_dnu

def linearize_jax_ctcs(fcn, constraints, n):

    def wrapped_fcn(t, z, nu, params):
        constr = jnp.concatenate([constraint.fcn(t, z[:n], nu, params) for constraint in constraints.get(ct=1)])
        f_val = jnp.concatenate([fcn(t, z[:n], nu, params), jnp.maximum(1.0*constr, 0.0)])
        return f_val

    dfcn_dz = jax.jacfwd(wrapped_fcn, argnums=1)
    dfcn_dnu = jax.jacfwd(wrapped_fcn, argnums=2)
    f = jax.jit(wrapped_fcn)
    
    return f, dfcn_dz, dfcn_dnu

    # TODO(Skye): Verify jax ctcs integration below or delete
    #     def augmented_fcn(t, z, nu, params):
    #         z_ctcs_idx = constraints.index_map.indices.z.ctcs
    #         f_val = fcn(z, nu, params)

    #         # TODO(Skye): remove and properly implement time dilation
    #         # Potentially add analytical jacobians for CTCS portion because we have those,
    #         # but use jax for constraint derivs
    #         if len(z_ctcs_idx) > 0:
    #             f_val = f_val.at[z_ctcs_idx].set(jnp.maximum(100 * constr_fcn(z, nu, params), 0.0))

    #         return f_val

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