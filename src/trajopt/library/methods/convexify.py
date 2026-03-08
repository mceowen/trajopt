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
        constr = jnp.concatenate([constraint.fcn(t, z[:n], nu, params) for constraint in constraints.get(ct=1)])
        f_val = jnp.concatenate([fcn(t, z[:n], nu, params), jnp.maximum(100*constr, 0.0)])
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