from collections.abc import Callable

import jax
import sympy as sp


# jax autodiff for affine approximations
def linearize_jax(fcn: Callable) -> tuple[Callable, Callable, Callable]:
    """Build JIT-compiled callables for fcn and its Jacobians w.r.t. z and nu."""
    dfcn_dz = jax.jit(jax.jacfwd(fcn, argnums=0))
    dfcn_dnu = jax.jit(jax.jacfwd(fcn, argnums=1))
    f = jax.jit(fcn)

    return f, dfcn_dz, dfcn_dnu


def hessian_jax(fcn):

    d2fcn_dz2 = jax.jit(jax.hessian(fcn, argnums=0))
    d2fcn_dnu2 = jax.jit(jax.hessian(fcn, argnums=1))
    d2fcn_dzdnu = jax.jit(jax.jacfwd(jax.jacrev(fcn, argnums=1), argnums=0))

    return d2fcn_dz2, d2fcn_dnu2, d2fcn_dzdnu

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
