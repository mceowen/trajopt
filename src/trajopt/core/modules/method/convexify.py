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
                    fcn_jit, dfcn_dz_jit, dfcn_du_jit = linearize_jax(constraint.fcn, problem)

                    def affine_approximation(t, z, nu, f=fcn_jit, dfdz=dfcn_dz_jit, dfdnu=dfcn_du_jit):
                        return f(z, nu), dfdz(z, nu), dfdnu(z, nu)
                    
                    constraint.affine_approximation = affine_approximation
                
                elif constraint.sympy == 1:
                    fcn_sym, dfcn_dz_sym, dfcn_du_sym = linearize_sympy(constraint.fcn, problem)
                    def affine_approximation(t, z, nu, f=fcn_sym, dfdz=dfcn_dz_sym, dfdnu=dfcn_du_sym):
                        return f(z, nu), dfdz(z, nu), dfdnu(z, nu)
                    constraint.affine_approximation = affine_approximation

            elif constraint.auto_diff == 0:
                
                def affine_approximation(t, z, nu, cnst_obj=constraint):
                    return cnst_obj.analytical_affine_approximation(t, z, nu, problem)
                
                constraint.affine_approximation = affine_approximation

def convexify_costs(problem):
    mission = problem.mission
    
    for cost in mission.costs:
        if cost.convex == 0:
            if cost.auto_diff == 1:
                if cost.jax == 1:
                    fcn_jit, dfcn_dz_jit, dfcn_du_jit = linearize_jax(cost.func, problem)

                    def affine_approximation(t, z, nu, f=fcn_jit, dfdz=dfcn_dz_jit, dfdnu=dfcn_du_jit):
                            return f(t, z, nu), dfdz(t, z, nu), dfdnu(t, z, nu)
                    cost.affine_approximation = affine_approximation
                
                else:
                    fcn_sym, dfcn_dz_sym, dfcn_du_sym = linearize_sympy(cost.func, problem)
                    def affine_approximation(t, z, nu, f=fcn_sym, dfdz=dfcn_dz_sym, dfdnu=dfcn_du_sym):
                            return f(t, z, nu), dfdz(t, z, nu), dfdnu(t, z, nu)
                    cost.affine_approximation = affine_approximation
        else:
            def affine_approximation(t, z, nu, cost_obj=cost):
                return cost_obj.analytical_affine_approximation(t, z, nu, problem)
            
            cost.affine_approximation = affine_approximation

        # TODO: add more options here for convexification

# jax autodiff for affine approximations
def linearize_jax(fcn, problem):

    def wrapped_fcn(z, nu):
        return fcn(0, z, nu, problem)

    dfcn_dz = jax.jit(jax.jacrev(wrapped_fcn, argnums=0))
    dfcn_du = jax.jit(jax.jacrev(wrapped_fcn, argnums=1))
    f = jax.jit(wrapped_fcn)

    return f, dfcn_dz, dfcn_du

def linearize_jax_ctcs(fcn, problem):
    method = problem.method
    
    def wrapped_fcn(z, nu):
        model = problem.model

        constr = jnp.concatenate([model.nonconvex_inequality_constraints[i].fcn(0, z, nu, problem) for i in range(len(model.nonconvex_inequality_constraints))])

        f_val = jnp.concatenate([fcn(0, z[:model.n], nu, problem), jnp.square(jnp.maximum(method.weights["w_ctcs"] * constr, 0.0))])
        
        return f_val

    dfcn_dz = jax.jacrev(wrapped_fcn, argnums=0)
    dfcn_du = jax.jacrev(wrapped_fcn, argnums=1)
    f = jax.jit(wrapped_fcn)
    
    return f, dfcn_dz, dfcn_du
    

# PROTOTYPE 
def linearize_sympy(fcn, problem):
    z, nu = problem.method.z_init, problem.method.nu_init
    n = problem.model.n
    m = problem.model.m

    z_sym = sp.symbols(f"z0:{n}")
    nu_sym = sp.symbols(f"u0:{m}")
    fcn_sym = fcn(0, z_sym, nu_sym, problem)
    dfcn_dz_sym = sp.diff(fcn_sym, z_sym)
    dfcn_du_sym = sp.diff(fcn_sym, nu_sym)
    return fcn_sym, dfcn_dz_sym, dfcn_du_sym

def compute_ctcs_jacobians(t, z, nu, problem):
    mission = problem.mission
    model = problem.model
    method = problem.method

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