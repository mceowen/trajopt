import numpy as np
import sympy as sp
import jax
import jax.numpy as jnp
import cvxpy as cp

def convexify_constraints(problem):
    model = problem.model

    for constraint in model.constraints:
        if constraint.convex == 0:
            if constraint.auto_diff == 1:
                if constraint.jax == 1:
                    fcn_jit, dfcn_dz_jit, dfcn_du_jit = linearize_jax(constraint.func, problem)

                    def affine_approximation(ts, zs, us, f=fcn_jit, dfdz=dfcn_dz_jit, dfdu=dfcn_du_jit):
                        return f(ts, zs, us), dfdz(ts, zs, us), dfdu(ts, zs, us)
                    
                    constraint.affine_approximation = affine_approximation
                
                elif constraint.sympy == 1:
                    fcn_sym, dfcn_dz_sym, dfcn_du_sym = linearize_sympy(constraint.func, problem)
                    def affine_approximation(ts, zs, us, f=fcn_sym, dfdz=dfcn_dz_sym, dfdu=dfcn_du_sym):
                        return f(ts, zs, us), dfdz(ts, zs, us), dfdu(ts, zs, us)
                    constraint.affine_approximation = affine_approximation

            elif constraint.auto_diff == 0:
                
                def affine_approximation(ts, zs, us, cnst_obj=constraint):
                    return cnst_obj.analytical_affine_approximation(ts, zs, us, problem)
                
                constraint.affine_approximation = affine_approximation

def convexify_costs(problem):
    mission = problem.mission
    
    for cost in mission.costs:
        if cost.convex == 0:
            if cost.auto_diff == 1:
                if cost.jax == 1:
                    fcn_jit, dfcn_dz_jit, dfcn_du_jit = linearize_jax(cost.func, problem)

                    def affine_approximation(ts, zs, us, f=fcn_jit, dfdz=dfcn_dz_jit, dfdu=dfcn_du_jit):
                            return f(ts, zs, us), dfdz(ts, zs, us), dfdu(ts, zs, us)
                    cost.affine_approximation = affine_approximation
                
                else:
                    fcn_sym, dfcn_dz_sym, dfcn_du_sym = linearize_sympy(cost.func, problem)
                    def affine_approximation(ts, zs, us, f=fcn_sym, dfdz=dfcn_dz_sym, dfdu=dfcn_du_sym):
                            return f(ts, zs, us), dfdz(ts, zs, us), dfdu(ts, zs, us)
                    cost.affine_approximation = affine_approximation
        else:
            def affine_approximation(ts, zs, us, cost_obj=cost):
                return cost_obj.analytical_affine_approximation(ts, zs, us, problem)
            
            cost.affine_approximation = affine_approximation

        # TODO: add more options here for convexification

# jax autodiff for affine approximations
def linearize_jax(fcn, problem):

    def wrapped_fcn(zs, us):
        return fcn(0, zs, us, problem)

    dfcn_dz = jax.jit(jax.jacrev(wrapped_fcn, argnums=0))
    dfcn_du = jax.jit(jax.jacrev(wrapped_fcn, argnums=1))
    f = jax.jit(wrapped_fcn)

    return f, dfcn_dz, dfcn_du

# PROTOTYPE 
def linearize_sympy(fcn, problem):
    zs, us = problem.method.zs_init, problem.method.us_init
    n = problem.model.n
    m = problem.model.m

    zs_sym = sp.symbols(f"z0:{n}")
    us_sym = sp.symbols(f"u0:{m}")
    fcn_sym = fcn(0, zs_sym, us_sym, problem)
    dfcn_dz_sym = sp.diff(fcn_sym, zs_sym)
    dfcn_du_sym = sp.diff(fcn_sym, us_sym)
    return fcn_sym, dfcn_dz_sym, dfcn_du_sym

def compute_ctcs_jacobians(ts, zs, us, problem):
    mission = problem.mission
    model = problem.model
    method = problem.method

    n = model.n
    n_ineq = mission.n_ineq

    # Evaluate linearized dynamics
    lin_dyn_info    = problem.lin_dyn(ts, zs[:n], us)
    f_xu = lin_dyn_info["fcn"]
    dfdx = lin_dyn_info["dfcn_dz"]
    dfdu = lin_dyn_info["dfcn_du"]

    # Evaluate linearized path constraints
    lin_constr_info = problem.lin_constr(ts, zs, us)
    g_xu = lin_constr_info["fcn"][0] * method.weights["w_ctcs"]
    dgdx = lin_constr_info["dfcn_dz"][0] * method.weights["w_ctcs"]
    dgdu = lin_constr_info["dfcn_du"][0]

    # Conditional constraint smoothing
    beta_dot = np.maximum(g_xu, 0.0) ** 2
    zeta = np.concatenate([f_xu, beta_dot])

    # Active constraint mask
    mask = g_xu > 0
    D = np.diag(mask.astype(float))

    # Constraint Jacobians
    dbetadot_dx = 2.0 * D @ np.diag(g_xu) @ dgdx
    dbetadot_du = 2.0 * D @ np.diag(g_xu) @ dgdu

    # Stack into full Jacobians
    dzeta_dz = np.block([
        [dfdx,               np.zeros((n, n_ineq))],
        [dbetadot_dx,        np.zeros((n_ineq, n_ineq))]
    ])

    dzeta_du = np.vstack([
        dfdu,
        dbetadot_du
    ])

    Ac = dzeta_dz
    Bc = dzeta_du
    fc = zeta

    return Ac, Bc, fc