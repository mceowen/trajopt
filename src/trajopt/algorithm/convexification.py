import numpy as np
import sympy as sp
import jax
import jax.numpy as jnp

# Functions are rough draft. Modify them later.


def compute_cost(ts, zs, us, problem):
    dcostdz = problem['lin_cost'](ts, zs, us)['dfcn_dz']
    dcostdu = problem['lin_cost'](ts, zs, us)['dfcn_du']
    cost    = problem['lin_cost'](ts, zs, us)['fcn']

    return dcostdz, dcostdu, cost


def compute_aero(ts, zs, us, problem):
    daero_dx = problem['lin_aero'](ts, zs, us)['dfcn_dz']
    daero_du = problem['lin_aero'](ts, zs, us)['dfcn_du']
    aero     = problem['lin_aero'](ts, zs, us)['fcn']

    return daero_dx, daero_du, aero


def compute_path_constraints(ts, zs, us, problem):
    dPdz = problem['lin_constr'](ts, zs, us)['dfcn_dz']
    dPdu = problem['lin_constr'](ts, zs, us)['dfcn_du']
    P    = problem['lin_constr'](ts, zs, us)['fcn']

    return dPdz, dPdu, P


def compute_linsys_continuous(ts, zs, us, problem):
    Ac = problem['lin_dyn'](ts, zs, us)['dfcn_dz']
    Bc = problem['lin_dyn'](ts, zs, us)['dfcn_du']
    fc = problem['lin_dyn'](ts, zs, us)['fcn']

    return Ac, Bc, fc


def compute_ctcs_jacobians(ts, zs, us, problem):
    params = problem["params"]
    n = params["n"]
    n_ineq = params["n_ineq"]

    # Evaluate linearized dynamics
    f_xu = problem["lin_dyn"]["fcn"](ts, zs, us)
    dfdx = problem["lin_dyn"]["dfcn_dz"](ts, zs, us)
    dfdu = problem["lin_dyn"]["dfcn_du"](ts, zs, us)

    # Evaluate linearized path constraints
    g_xu = problem["lin_constr"]["fcn"](ts, zs, us)
    dgdx = problem["lin_constr"]["dfcn_dz"](ts, zs, us)
    dgdu = problem["lin_constr"]["dfcn_du"](ts, zs, us)

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


def generate_jacobians(fcn_hdl, problem):

    # Extract params
    params = problem.get("params", problem)
    n = params["n"]
    m = params["m"]

    # Symbolic variables
    t = sp.Symbol("t")
    z = sp.Matrix(sp.symbols("z0:%d" % n))
    u = sp.Matrix(sp.symbols("u0:%d" % m))

    # Nonlinear function and Jacobians
    fcn_sym = sp.simplify(fcn_hdl(t, z, u))
    dfcn_dz_sym = sp.simplify(fcn_sym.jacobian(z))
    dfcn_du_sym = sp.simplify(fcn_sym.jacobian(u))

    # Turn symbolic expressions into callable functions
    dfdz_hdl = sp.lambdify((t, z, u), dfcn_dz_sym, modules='numpy')
    dfdu_hdl = sp.lambdify((t, z, u), dfcn_du_sym, modules='numpy')
    fcn_eval = sp.lambdify((t, z, u), fcn_sym, modules='numpy')

    def lin_hdl(t_val, z_val, u_val):
        z_arr = np.array(z_val).flatten()
        u_arr = np.array(u_val).flatten()
        return {
            "dfcn_dz": np.array(dfdz_hdl(t_val, z_arr, u_arr), dtype=float),
            "dfcn_du": np.array(dfdu_hdl(t_val, z_arr, u_arr), dtype=float),
            "fcn":     np.array(fcn_eval(t_val, z_arr, u_arr), dtype=float)
        }

    return lin_hdl


def generate_jacobians_jax(fcn_hdl, problem):
    
    params = problem.get("params", problem)
    n, m = params["n"], params["m"]

    # Wrap function to make t, z, u separate JAX arguments
    def wrapped(z_u, t):
        z = z_u[:n]
        u = z_u[n:]
        return fcn_hdl(t, z, u)

    # Jacobians w.r.t. z and u
    def lin_hdl(t, z, u):
        z = jnp.atleast_1d(jnp.asarray(z))
        u = jnp.atleast_1d(jnp.asarray(u))
        z_u = jnp.concatenate([z, u])

        J = jax.jacobian(wrapped)(z_u, t)
        dfcn_dz = J[:, :n]
        dfcn_du = J[:, n:]
        f = fcn_hdl(t, z, u)

        return {
            "dfcn_dz": dfcn_dz,
            "dfcn_du": dfcn_du,
            "fcn": f
        }

    return lin_hdl

def generate_jacobians2(func_nl, params):
    """
    Generate symbolic Jacobian function handles from a nonlinear symbolic function.

    Parameters:
        func_nl : function handle returning sympy Matrix, i.e. func_nl(t, z, u)
        params  : dict with 'n' (state dimension) and 'm' (control dimension)

    Returns:
        lin_hdl : dict with keys:
                    'fcn'      - original symbolic function
                    'dfcn_dx'  - ∂f/∂z function handle
                    'dfcn_du'  - ∂f/∂u function handle
    """
    n = params["n"]
    m = params["m"]

    # Define symbolic variables
    t = sp.Symbol('t')
    z = sp.Matrix(sp.symbols(f'z0:{n}')).reshape(n, 1)
    u = sp.Matrix(sp.symbols(f'u0:{m}')).reshape(m, 1)

    # Evaluate symbolic nonlinear function
    f_sym = sp.simplify(func_nl(t, z, u))
    assert isinstance(f_sym, sp.Matrix), "func_nl must return a sympy Matrix"

    # Compute Jacobians with respect to z (state)
    dfdx_sym = sp.zeros(len(f_sym), n)
    for i in range(len(f_sym)):
        for j in range(n):
            dfdx_sym[i, j] = sp.simplify(sp.diff(f_sym[i], z[j]))

    # Compute Jacobians with respect to u (control)
    dfdu_sym = sp.zeros(len(f_sym), m)
    for i in range(len(f_sym)):
        for j in range(m):
            dfdu_sym[i, j] = sp.simplify(sp.diff(f_sym[i], u[j]))

    # Create callable function handles
    dfdx_hdl = sp.lambdify((t, z, u), dfdx_sym, modules='numpy')
    dfdu_hdl = sp.lambdify((t, z, u), dfdu_sym, modules='numpy')

    lin_hdl = {
        "fcn": func_nl,
        "dfcn_dx": dfdx_hdl,
        "dfcn_du": dfdu_hdl
    }

    return lin_hdl