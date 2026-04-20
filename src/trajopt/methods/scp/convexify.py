import sympy as sp
import jax
import jax.numpy as jnp

# jax autodiff for affine approximations
def linearize_jax(fcn):

    dfcn_dz = jax.jit(jax.jacfwd(fcn, argnums=0))
    dfcn_dnu = jax.jit(jax.jacfwd(fcn, argnums=1))
    f = jax.jit(fcn)

    return f, dfcn_dz, dfcn_dnu

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