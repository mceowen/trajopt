import numpy as np
import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from scipy.integrate import solve_ivp
import trajopt.algorithm.convexification as convexify
import time

def set_ltv_indices(params):
    """
    Function to set Linear Time Varying (LTV) indices and initialize arrays.

    Parameters:
    params (dict): Dictionary containing parameters.

    Returns:
    dict: Updated params with LTV indices and initialized arrays.
    """
    params['z_ind']     = np.arange(0, params['nz'])
    params['Ak_ind']    = np.arange(params['z_ind'][-1] + 1, params['z_ind'][-1] + 1 + params['nz']**2 )
    params['Bk_ind']    = np.arange(params['Ak_ind'][-1] + 1, params['Ak_ind'][-1] + 1 + params['nz'] * params['m'] )
    params['Bkp_ind']   = np.arange(params['Bk_ind'][-1] + 1, params['Bk_ind'][-1] + 1 + params['nz'] * params['m'] )
    params['Sk_ind']    = np.arange(params['Bkp_ind'][-1] + 1, params['Bkp_ind'][-1] +
                                    1 + params['nz'] )

    params['Ak']        = np.zeros((params['N'] - 1,params['nz'], params['nz']))
    params['Bk']        = np.zeros((params['N'] - 1, params['nz'], params['m']))
    params['Bkp']       = np.zeros((params['N'] - 1, params['nz'], params['m']))
    params['Sk']        = np.zeros((params['N'] - 1, params['nz']))

    params['lds0_size'] = params['Sk_ind'][-1] + 1
    params['lds0']      = np.zeros( params['lds0_size'] )

    params['lds0'][params['Ak_ind']] = np.reshape(np.eye(params['nz']), -1)
    params['N_dens']    = 20

    # convert indeces to jax arrays for jax discretization option
    params['lds0_size_jax'] = int(params['lds0_size'])
    params['z_ind_jax']     = jnp.asarray(params['z_ind'])  
    params['Ak_ind_jax']    = jnp.asarray(params['Ak_ind'])
    params['Bk_ind_jax']    = jnp.asarray(params['Bk_ind'])
    params['Bkp_ind_jax']   = jnp.asarray(params['Bkp_ind'])
    params['Sk_ind_jax']    = jnp.asarray(params['Sk_ind'])

    return params

# Compute exact discretization for linear dynamic system
def discretize_inv_foh(zs_ref, us_ref, dts_ref, problem):
    params = problem['params']
    N = params['N']

    traj_minus_data = {'zs_minus': [zs_ref[0]]}

    # Precompute
    eye_flat = np.eye(params['n']).ravel()

    # Stack dynamics
    lds0_stack = []
    for k in range(N - 1):
        params['lds0'][params['z_ind']] = zs_ref[k]
        params['lds0'][params['Ak_ind']] = eye_flat
        lds0_stack.append(params['lds0'].copy())

    lds0_stack = np.concatenate(lds0_stack)

    def derivs_step(tau, lds):
        return RHS_ltv(tau, lds, us_ref, dts_ref, problem)

    sol = solve_ivp(derivs_step, [0, 1], lds0_stack, method='RK45', atol=1e-6, rtol=1e-6)

    lds_end = sol.y[:, -1]  # shape: (total_state_size,)

    assert lds_end.shape[0] == (N - 1) * params['lds0_size']

    Ak  = np.zeros((N - 1, params['n'], params['n']))
    Bk  = np.zeros((N - 1, params['n'], params['m']))
    Bkp = np.zeros((N - 1, params['n'], params['m']))
    Sk  = np.zeros((N - 1, params['n']))

    for k in range(N - 1):
        base    = k * params['lds0_size']
        traj_minus_data['zs_minus'].append(lds_end[base + params['z_ind']])

        Ak_bar  = lds_end[base + params['Ak_ind']].reshape(params['n'], params['n'])
        Bk_bar  = lds_end[base + params['Bk_ind']].reshape(params['n'], params['m'])
        Bkp_bar = lds_end[base + params['Bkp_ind']].reshape(params['n'], params['m'])
        Sk_bar  = lds_end[base + params['Sk_ind']].reshape(params['n'])

        Ak[k]   = Ak_bar
        Bk[k]   = Ak_bar @ Bk_bar
        Bkp[k]  = Ak_bar @ Bkp_bar
        Sk[k]   = Ak_bar @ Sk_bar

    zs_minus    = np.array(traj_minus_data['zs_minus'])

    return Ak, Bk, Bkp, Sk, zs_minus


# Integrate linear system
def RHS_ltv(tau, lds, us_ref, dts_ref, problem):

    # Initialize
    lds_dot         = np.zeros_like(lds)
    params          = problem['params']
    N               = params['N']

    # Extract times and FOH control input
    Om_k            = 1 - tau
    Om_kp           = tau

    nrows, ncols    = us_ref.shape
    rows            = nrows if nrows > ncols else ncols

    v_1             = Om_k * np.ones((rows, 1))
    v_2             = Om_kp * np.ones((rows - 1, 1))
    Om              = np.diagflat(v_1) + np.diagflat(v_2, 1)

    u               = Om @ us_ref

    for k in range(N - 1):
        dts_k       = dts_ref[k]

        # Extract state info
        x           = lds[ k * params['lds0_size'] + params['z_ind'] ]

        # Extract continuous time Jacobians
        Ac, Bc, fc  = convexify.compute_linsys_continuous(tau, x, u[k], problem)

        # Extract STM
        Phi_tau     = lds[ k * params['lds0_size'] + params['Ak_ind'] ].reshape(params['n'], params['n'])

        # Construct Jacobians w.r.t. tau
        f_tau       = dts_k * fc
        A_tau       = dts_k * Ac
        B_tau       = dts_k * Om_k * Bc
        Bp_tau      = dts_k * Om_kp * Bc
        S_tau       = fc

        Phi_tau_inv = np.linalg.pinv(Phi_tau)

        # Construct derivatives
        x_dot       = f_tau
        A_tau_dot   = A_tau @ Phi_tau
        B_tau_dot   = Phi_tau_inv @ B_tau
        Bp_tau_dot  = Phi_tau_inv @ Bp_tau
        S_tau_dot   = Phi_tau_inv @ S_tau

        # Setup linear system properly
        lds_dot[ k * params['lds0_size'] + params['z_ind']]     = x_dot
        lds_dot[ k * params['lds0_size'] + params['Ak_ind']]    = A_tau_dot.flatten()
        lds_dot[ k * params['lds0_size'] + params['Bk_ind']]    = B_tau_dot.flatten()
        lds_dot[ k * params['lds0_size'] + params['Bkp_ind']]   = Bp_tau_dot.flatten()
        lds_dot[ k * params['lds0_size'] + params['Sk_ind']]    = S_tau_dot

    return lds_dot

def jit_jax_discretization(problem):
    params = problem['params']
    
    n = int(params['nz'])
    m = int(params['m'])
    N = int(params['N'])

    # define static indices for stacked RHS vector 
    Ak_ind0   = n
    Bk_ind0   = Ak_ind0  + n*n
    Bkp_ind0  = Bk_ind0  + n*m
    Sk_ind0   = Bkp_ind0 + n*m

    # pull ltv dynamics
    lin_sys = problem['lin_dyn']

    # nsub defines the number of sub *nodes* between knot points
    nsub_nodes = 30
    dt_sub = 1.0 / (nsub_nodes + 1)
    t = jnp.linspace(0.0, 1.0, nsub_nodes + 2)

    # packs the derivative of stacked RHS vector for node k
    def pack_lds_dot(tau, lds_k, us_k, us_kp, dts_k):
        a = 1 - tau
        b = tau

        x       = lds_k[         : Ak_ind0]
        phi_a   = lds_k[Ak_ind0  : Bk_ind0].reshape((n, n))
        phi_b_m = lds_k[Bk_ind0  : Bkp_ind0].reshape((n, m))
        phi_b_p = lds_k[Bkp_ind0 : Sk_ind0].reshape((n, m))
        phi_s   = lds_k[Sk_ind0  : ]

        u = a * us_k + b * us_kp
        sigma = dts_k

        lin_info    = lin_sys(tau, x, u)

        Ac = lin_info['dfcn_dz']
        Bc = lin_info['dfcn_du']
        fc = lin_info['fcn']

        P1_dot = (sigma * fc)
        P2_dot = (sigma * Ac @ phi_a).reshape((n*n,))
        P3_dot = (sigma * Ac @ phi_b_m + sigma * Bc * a).reshape((n*m,))
        P4_dot = (sigma * Ac @ phi_b_p + sigma * Bc * b).reshape((n*m,))
        P5_dot = (sigma * Ac @ phi_s   + fc)

        return jnp.concatenate([P1_dot, P2_dot, P3_dot, P4_dot, P5_dot])

    # rk4 single step function for jax integration
    def rk4_step_jax(tau, lds, us_k, us_kp, dts_k):
        k1 = pack_lds_dot(tau, lds, us_k, us_kp, dts_k)
        k2 = pack_lds_dot(tau + dt_sub / 2, lds + (dt_sub / 2) * k1, us_k, us_kp, dts_k)
        k3 = pack_lds_dot(tau + dt_sub / 2, lds + (dt_sub / 2) * k2, us_k, us_kp, dts_k)
        k4 = pack_lds_dot(tau + dt_sub, lds + dt_sub * k3, us_k, us_kp, dts_k)
        
        lds_next = lds + (dt_sub / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return lds_next, None

    rk4_step_jax_jit = jax.jit(rk4_step_jax)

    # initilize stacked propagation vector  
    def pack_lds0(zs_k):
        P1 = zs_k
        P2 = jnp.eye(n).reshape(n*n)
        P3 = jnp.zeros(n*m)
        P4 = jnp.zeros(n*m)
        P5 = jnp.zeros(n)

        return jnp.concatenate([P1, P2, P3, P4, P5])

    # unpacks stacked propagation vector to correct shapes
    def unpack_ldsf(ldsf_k):
        zs_minus_k = ldsf_k[ : Ak_ind0]
        A_jax_k    = ldsf_k[Ak_ind0  : Bk_ind0].reshape((n, n))
        B_jax_k    = ldsf_k[Bk_ind0  : Bkp_ind0].reshape((n, m))
        Bp_jax_k   = ldsf_k[Bkp_ind0 : Sk_ind0].reshape((n, m))
        S_jax_k    = ldsf_k[Sk_ind0  : ]

        return (A_jax_k, B_jax_k, Bp_jax_k, S_jax_k, zs_minus_k)

    # propagation function for node k
    def propagate_k(k, zs_ref, us_ref, dts_ref):
        zs_k  = zs_ref[k]
        us_k  = us_ref[k]
        us_kp = us_ref[k+1]
        dts_k = dts_ref[k]

        # stack z, A, B, Bp, S for multiple shooting propagation
        lds0_k = pack_lds0(zs_k)

        # propagate stacked vector
        def rk4_step_jax_partial(lds, tau):
            return rk4_step_jax_jit(tau, lds, us_k, us_kp, dts_k)

        ldsf_k, _ = jax.lax.scan(rk4_step_jax_partial, lds0_k, t[:-1])

        # unpack the stacked vector back into appropriate shapes
        return unpack_ldsf(ldsf_k)
    
    propagate = jax.jit(jax.vmap(propagate_k, in_axes=(0, None, None, None)))

    problem['propagate'] = propagate

    return problem

# inverse free discretization with jax
def discretize_inv_free_jax(zs_ref_np, us_ref_np, dts_ref_np, problem):

    params = problem['params']

    # get jitted jax propagation
    problem['propagate']

    # convert numpy arrays to jax
    zs_ref = jnp.asarray(zs_ref_np)
    us_ref = jnp.asarray(us_ref_np)
    dts_ref = jnp.asarray(dts_ref_np)

    # call jitted propagator for each node
    ks = jnp.arange(params['N'] - 1)
    A_jax, B_jax, Bp_jax, S_jax, zs_minus = problem['propagate'](ks, zs_ref, us_ref, dts_ref)

    zs_ref_0 = zs_ref[[0], :]
    
    return np.asarray(A_jax), np.asarray(B_jax), np.asarray(Bp_jax), np.asarray(S_jax), np.asarray(jnp.vstack([zs_ref_0, zs_minus]))

def compute_linsys_discrete(zs_ref, us_ref, dts_ref, problem):
    """
    Compute the linear system in discrete form.

    Parameters:
    zs_ref (numpy.ndarray): Reference state trajectory.
    us_ref (numpy.ndarray): Reference control trajectory.
    dts_ref (numpy.ndarray): Time steps.
    problem (dict): Dictionary containing problem parameters.

    Returns:
    tuple: Ak, Bk, Bkp, Sk, zs_minus
    """
    if problem['params']['bools']['ctcs']:
        Ak, Bk, Bkp, Sk, zs_minus = discretize_ctcs(zs_ref, us_ref, dts_ref, problem)
    elif problem['params']['bools']['jax_dyn']:
        Ak, Bk, Bkp, Sk, zs_minus = discretize_inv_free_jax(zs_ref, us_ref, dts_ref, problem)
    else:
        Ak, Bk, Bkp, Sk, zs_minus = discretize_inv_foh(zs_ref, us_ref, dts_ref, problem)
    
    return Ak, Bk, Bkp, Sk, zs_minus


def discretize_ctcs(zs_ref, us_ref, dts_ref, problem):
    """
    Compute exact discretization for linear dynamic system.

    Parameters:
    zs_ref (numpy.ndarray): Reference state trajectory.
    us_ref (numpy.ndarray): Reference control trajectory.
    dts_ref (numpy.ndarray): Time steps.
    problem (dict): Dictionary containing problem parameters.

    Returns:
    tuple: Ak, Bk, Bkp, Sk, zs_minus
    """
    params = problem['params']
    N = params['N']

    traj_minus_data = {'zs_minus': [zs_ref[0]]}

    # Setup LTV system dynamics
    lds0_stack = []
    for k in range(N - 1):
        params['lds0'][params['z_ind']] = zs_ref[k]
        params['lds0'][params['Ak_ind']] = np.reshape(np.eye(params['nz']), -1)
        lds0_stack.append(params['lds0'].copy())

    lds0_stack = np.hstack(lds0_stack)

    def derivs_step(tau, lds):
        return RHS_ltv_ctcs(tau, lds, us_ref, dts_ref, problem)

    sol = solve_ivp(derivs_step, [0, 1], lds0_stack, atol=1E-12, rtol=1E-12)
    lds_out_stack = sol.y

    Ak  = np.zeros((N-1, params['nz'],  params['nz']))
    Bk  = np.zeros((N-1, params['nz'],  params['m']))
    Bkp = np.zeros((N-1, params['nz'],   params['m']))
    Sk  = np.zeros((N-1, params['nz']))

    # Extract dense values
    for k in range(N - 1):
        lds_end = lds_out_stack[:, -1]
        traj_minus_data['zs_minus'].append(
            lds_end[k * params['lds0_size'] + params['z_ind']])

        # Reshape matrices
        Ak_bar  = np.reshape(lds_end[k * params['lds0_size'] + params['Ak_ind']],
                            (params['nz'], params['nz']))
        Bk_bar  = np.reshape(lds_end[k * params['lds0_size'] + params['Bk_ind']],
                            (params['nz'], params['m']))
        Bkp_bar = np.reshape(lds_end[k * params['lds0_size'] + params['Bkp_ind']],
                             (params['nz'], params['m']))
        Sk_bar  = lds_end[k * params['lds0_size'] + params['Sk_ind']]

        # Fill in the next STM
        Ak[k]   = Ak_bar
        Bk[k]   = Ak_bar @ Bk_bar
        Bkp[k]  = Ak_bar @ Bkp_bar
        Sk[k]   = Ak_bar @ Sk_bar

    # Extract x_ref_minus traj (from integration)
    zs_minus    = np.array(traj_minus_data['zs_minus'])

    return Ak, Bk, Bkp, Sk, zs_minus


def RHS_ltv_ctcs(tau, lds, us_ref, dts_ref, problem):
    """
    Integrate linear system.

    Parameters:
    tau (float): Current time step.
    lds (numpy.ndarray): Linear dynamic system state.
    us_ref (numpy.ndarray): Reference control trajectory.
    dts_ref (numpy.ndarray): Time steps.
    problem (dict): Dictionary containing problem parameters.

    Returns:
    numpy.ndarray: Derivative of the linear dynamic system state.
    """
    lds_dot = np.zeros_like(lds)
    params = problem['params']
    N = params['N']

    Om_k = 1 - tau
    Om_kp = tau

    Om = np.diag(Om_k * np.ones(us_ref.shape[0])) + np.diag(Om_kp * np.ones(us_ref.shape[0] - 1), 1)

    u = Om @ us_ref

    for k in range(N - 1):
        dts_k = dts_ref[k]
        x = lds[k * params['lds0_size'] + params['z_ind']]

        Ac, Bc, fc = convexify.compute_ctcs_jacobians(tau, x, u[k, :], problem)

        Phi_tau = np.reshape(lds[k * params['lds0_size'] + params['Ak_ind']],
                             (params['nz'], params['nz']))

        f_tau = dts_k * fc
        A_tau = dts_k * Ac
        B_tau = dts_k * Om_k * Bc
        Bp_tau = dts_k * Om_kp * Bc
        S_tau = fc

        Phi_tau_inv = np.linalg.inv(Phi_tau)

        x_dot = f_tau
        A_tau_dot = A_tau @ Phi_tau
        B_tau_dot = Phi_tau_inv @ B_tau
        Bp_tau_dot = Phi_tau_inv @ Bp_tau
        S_tau_dot = Phi_tau_inv @ S_tau

        lds_dot[k * params['lds0_size'] + params['z_ind']] = x_dot
        lds_dot[k * params['lds0_size'] + params['Ak_ind']] = np.reshape(A_tau_dot, -1)
        lds_dot[k * params['lds0_size'] + params['Bk_ind']] = np.reshape(B_tau_dot, -1)
        lds_dot[k * params['lds0_size'] + params['Bkp_ind']] = np.reshape(Bp_tau_dot, -1)
        lds_dot[k * params['lds0_size'] + params['Sk_ind']] = S_tau_dot

    return lds_dot

# Example usage
if __name__ == "__main__":

     # Define dummy data for testing
    params = {
        'nz': 3,
        'm': 2,
        'N': 10
    }

    updated_params = set_ltv_indices(params)
    print(updated_params)

    # Define dummy inputs for testing
    zs_ref = np.random.rand(3, 4)
    us_ref = np.random.rand(4, 3)
    dts_ref = np.random.rand(3)
    problem = {
        'params': {
            'N': 4,
            'n': 3,
            'm': 3,
            'lds0_size': 9,
            'z_ind': slice(0, 3),
            'Ak_ind': slice(3, 6),
            'Bk_ind': slice(6, 9),
            'Bkp_ind': slice(9, 12),
            'Sk_ind': slice(12, 15),
            'lds0': np.zeros(15),
            'bools': {'ode_fixed_dt': False}
        }
    }

    Ak, Bk, Bkp, Sk, zs_minus = discretize_inv_foh(zs_ref, us_ref, dts_ref, problem)
    print(f"Ak: {Ak}")
    print(f"Bk: {Bk}")
    print(f"Bkp: {Bkp}")
    print(f"Sk: {Sk}")
    print(f"zs_minus: {zs_minus}")