import numpy as np
import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from scipy.integrate import solve_ivp
import trajopt.library.methods.convexify as convexify
import trajopt.utils.tools as tools

def set_ltv_indices(problem, method):
    """
    Function to set Linear Time Varying (LTV) indices and initialize arrays.

    Parameters:
    params (dict): Dictionary containing parameters.

    Returns:
    dict: Updated params with LTV indices and initialized arrays.
    """

    nz = problem.index_map.n['z']
    n_nu = problem.index_map.n['control']
    N = method.index_map.N['N']

    method.z_ind     = np.arange(0, nz)
    method.Ak_ind    = np.arange(method.z_ind[-1] + 1, method.z_ind[-1] + 1 + nz**2 )
    method.Bk_ind    = np.arange(method.Ak_ind[-1] + 1, method.Ak_ind[-1] + 1 + nz*n_nu )
    method.Bkp_ind   = np.arange(method.Bk_ind[-1] + 1, method.Bk_ind[-1] + 1 + nz*n_nu )
    method.Sk_ind    = np.arange(method.Bkp_ind[-1] + 1, method.Bkp_ind[-1] + 1 + nz )

    method.Ak        = np.zeros((method.index_map.N['N'] - 1, nz, nz))
    method.Bk        = np.zeros((method.index_map.N['N'] - 1, nz, n_nu))
    method.Bkp       = np.zeros((method.index_map.N['N'] - 1, nz, n_nu))
    method.Sk        = np.zeros((method.index_map.N['N'] - 1, nz, 1))

    method.lds0_size = method.Sk_ind[-1] + 1
    method.lds0      = np.zeros( method.lds0_size )

    method.lds0[method.Ak_ind] = np.reshape(np.eye(nz), -1)

    # convert indeces to jax arrays for jax discretize option
    method.lds0_size_jax = int(method.lds0_size)
    method.z_ind_jax     = jnp.asarray(method.z_ind)  
    method.Ak_ind_jax    = jnp.asarray(method.Ak_ind)
    method.Bk_ind_jax    = jnp.asarray(method.Bk_ind)
    method.Bkp_ind_jax   = jnp.asarray(method.Bkp_ind)
    method.Sk_ind_jax    = jnp.asarray(method.Sk_ind)

def compute_nonconvex_constraints(t_ref, z_ref, u_ref, problem, method):
    n_ineq  = problem.index_map.n['nonconvex_inequality']
    n_x     = problem.index_map.n['state']
    n_nu    = problem.index_map.n['control']
    N       = method.index_map.N['N']
        
    t_jax = jnp.asarray(t_ref)
    z_jax = jnp.asarray(z_ref)
    nu_jax = jnp.asarray(u_ref)

    params = problem.params
    params_jax = tools.recursive_to_dict(params)
    
    # Preallocate stacked arrays
    g    = np.zeros((N, n_ineq))
    dgdz = np.zeros((N, n_ineq, n_x))
    dgdnu = np.zeros((N, n_ineq, n_nu))
    
    # Evaluate constraints at each timestep
    for k in range(N):
        tk = t_jax[k]
        zk = z_jax[k, :n_x]
        uk = nu_jax[k]
        
        col_start = 0
        for constraint in problem.constraints.get(ct=0, type="nonconvex_inequality"):
            col_end = col_start + constraint.dimension
            
            f, dfcn_dz, dfcn_du            = constraint.g_aff(tk, zk, uk, params_jax)

            g_k = np.asarray(f)
            dgdz_k = np.asarray(dfcn_dz)
            dgdnu_k = np.asarray(dfcn_du)

            g[k, col_start:col_end]        = g_k
            dgdz[k, col_start:col_end, :]  = dgdz_k
            dgdnu[k, col_start:col_end, :] = dgdnu_k

            # # condition constraints by row norms
            # row_norms = np.linalg.norm(np.hstack([dgdz_k, dgdnu_k]), axis=1)

            # # avoid division by zero
            # row_norms[row_norms == 0] = 1.0
            # g[k, col_start:col_end] /= row_norms
            # dgdz[k, col_start:col_end, :] /= row_norms[:, np.newaxis]
            # dgdnu[k, col_start:col_end, :] /= row_norms[:, np.newaxis]
            
            col_start = col_end
    
    return g, dgdz, dgdnu

def compute_nonconvex_costs(t_ref, z_ref, u_ref, problem, method):

    n_x = problem.index_map.n['state']
    n_nu = problem.index_map.n['control']
    N = method.index_map.N['N']
    
    t_jax = jnp.asarray(t_ref)
    z_jax = jnp.asarray(z_ref)
    nu_jax = jnp.asarray(u_ref)

    params = problem.params
    params_jax = tools.recursive_to_dict(params)
    
    # preallocate stacked arrays (cost per timestep)
    cost = np.zeros((N, 1))
    dcostdz = np.zeros((N, 1, n_x))
    dcostdnu = np.zeros((N, 1, n_nu))

    # evaluate costs at each timestep
    for cost_object in problem.costs.get(ct=0, type="nonconvex"):
        for k in range(N-1):
            tk = t_jax[k]
            zk = z_jax[k]
            uk = nu_jax[k]
            
            f, dfcn_dz, dfcn_du = cost_object.g_aff(tk, zk, uk, params_jax)
            cost[k, 0] += np.asarray(f)
            dcostdz[k, 0, :] += np.asarray(dfcn_dz).flatten()
            dcostdnu[k, 0, :] += np.asarray(dfcn_du).flatten()
            
    for cost_object in problem.costs.get(type="nonconvex_terminal"):
        tk = t_jax[-1]
        zk = z_jax[-1]
        uk = nu_jax[-1]
        
        f, dfcn_dz, dfcn_du = cost_object.g_aff(tk, zk, uk, params_jax)
        cost[-1, 0]        += np.asarray(f)
        dcostdz[-1, 0, :]  += np.asarray(dfcn_dz).flatten()
        dcostdnu[-1, 0, :] += np.asarray(dfcn_du).flatten()
    
    return cost, dcostdz, dcostdnu

def compile_jax_discretization(problem, method):
    nz = problem.index_map.n['z']
    n_nu = problem.index_map.n['control']

    # define static indices for stacked RHS vector 
    Ak_ind0   = nz
    Bk_ind0   = Ak_ind0  + nz*nz
    Bkp_ind0  = Bk_ind0  + nz*n_nu
    Sk_ind0   = Bkp_ind0 + nz*n_nu

    params = problem.params
    params_jax = tools.recursive_to_dict(params)

    # pull ltv dynamics
    lin_dyn = problem.constraints.get(type='dynamics')[0].lin_dyn

    # nsub defines the number of sub *nodes* between knot points
    nsub_nodes = 100
    dt_sub = 1.0 / (nsub_nodes + 1)
    t = jnp.linspace(0.0, 1.0, nsub_nodes + 2)

    # packs the derivative of stacked RHS vector for node k
    def pack_lds_dot(tau, lds_k, nu_k, nu_kp, dt_k, params_jax):

        x       = lds_k[         : Ak_ind0]
        phi_a   = lds_k[Ak_ind0  : Bk_ind0].reshape((nz, nz))
        phi_b_m = lds_k[Bk_ind0  : Bkp_ind0].reshape((nz, n_nu))
        phi_b_p = lds_k[Bkp_ind0 : Sk_ind0].reshape((nz, n_nu))
        phi_s   = lds_k[Sk_ind0  : ]

        a = 1 - tau
        b = tau
        u = a * nu_k + b * nu_kp
        sigma = dt_k

        fc, Ac, Bc    = lin_dyn(tau, x, u, params_jax)

        P1_dot = (sigma * fc)
        P2_dot = (sigma * Ac @ phi_a).reshape((nz*nz,))
        P3_dot = (sigma * Ac @ phi_b_m + sigma * Bc * a).reshape((nz*n_nu,))
        P4_dot = (sigma * Ac @ phi_b_p + sigma * Bc * b).reshape((nz*n_nu,))
        P5_dot = (sigma * Ac @ phi_s   + fc)

        return jnp.concatenate([P1_dot, P2_dot, P3_dot, P4_dot, P5_dot])

    # rk4 single step function for jax integration
    def rk4_step_jax(tau, lds, nu_k, nu_kp, dt_k, params_jax):
        
        k1 = pack_lds_dot(tau, lds, nu_k, nu_kp, dt_k, params_jax)
        k2 = pack_lds_dot(tau + dt_sub / 2, lds + (dt_sub / 2) * k1, nu_k, nu_kp, dt_k, params_jax)
        k3 = pack_lds_dot(tau + dt_sub / 2, lds + (dt_sub / 2) * k2, nu_k, nu_kp, dt_k, params_jax)
        k4 = pack_lds_dot(tau + dt_sub, lds + dt_sub * k3, nu_k, nu_kp, dt_k, params_jax)
        
        lds_next = lds + (dt_sub / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return lds_next, None

    rk4_step_jax_jit = jax.jit(rk4_step_jax)

    # initilize stacked propagation vector  
    def pack_lds0(z_k):
        P1 = z_k
        P2 = jnp.eye(nz).reshape(nz*nz)
        P3 = jnp.zeros(nz*n_nu)
        P4 = jnp.zeros(nz*n_nu)
        P5 = jnp.zeros(nz)

        return jnp.concatenate([P1, P2, P3, P4, P5])

    # unpacks stacked propagation vector to correct shapes
    def unpack_ldsf(ldsf_k):
        z_minus_k = ldsf_k[ : Ak_ind0]
        A_jax_k    = ldsf_k[Ak_ind0  : Bk_ind0].reshape((nz, nz))
        B_jax_k    = ldsf_k[Bk_ind0  : Bkp_ind0].reshape((nz, n_nu))
        Bp_jax_k   = ldsf_k[Bkp_ind0 : Sk_ind0].reshape((nz, n_nu))
        S_jax_k    = ldsf_k[Sk_ind0  : ]

        return (A_jax_k, B_jax_k, Bp_jax_k, S_jax_k, z_minus_k)

    # propagation function for node k
    def propagate_k(k, z_ref, nu_ref, dt_ref, params_jax):

        z_k  = z_ref[k]
        nu_k  = nu_ref[k]
        nu_kp = nu_ref[k+1]
        dt_k = dt_ref[k]

        # stack z, A, B, Bp, S for multiple shooting propagation
        lds0_k = pack_lds0(z_k)

        # propagate stacked vector
        def rk4_step_jax_partial(lds, tau):
            return rk4_step_jax_jit(tau, lds, nu_k, nu_kp, dt_k, params_jax)

        ldsf_k, _ = jax.lax.scan(rk4_step_jax_partial, lds0_k, t[:-1])

        # unpack the stacked vector back into appropriate shapes
        return unpack_ldsf(ldsf_k)
    
    propagate = jax.jit(jax.vmap(propagate_k, in_axes=(0, None, None, None, None)))

    method.propagate_discretization_jax = propagate

def compile_jax_discretization_bwd(problem, method):
    nz = problem.index_map.n['z']
    n_nu = problem.index_map.n['control']

    # define static indices for stacked RHS vector 
    Ak_ind0   = nz
    Bk_ind0   = Ak_ind0  + nz*nz
    Bkp_ind0  = Bk_ind0  + nz*n_nu
    Sk_ind0   = Bkp_ind0 + nz*n_nu

    params = problem.params

    # pull ltv dynamics
    lin_dyn = problem.constraints.get(type='dynamics')[0].lin_dyn

    # nsub defines the number of sub *nodes* between knot points
    nsub_nodes = 20
    dt_sub = -1.0 / (nsub_nodes + 1)
    t = jnp.linspace(0.0, 1.0, nsub_nodes + 2)

    # packs the derivative of stacked RHS vector for node k
    def pack_lds_dot(tau, lds_k, nu_k, nu_kp, dt_k, params_jax):

        x       = lds_k[         : Ak_ind0]
        phi_a   = lds_k[Ak_ind0  : Bk_ind0].reshape((nz, nz))
        phi_b_m = lds_k[Bk_ind0  : Bkp_ind0].reshape((nz, n_nu))
        phi_b_p = lds_k[Bkp_ind0 : Sk_ind0].reshape((nz, n_nu))
        phi_s   = lds_k[Sk_ind0  : ]

        a = 1 - tau
        b = tau
        u = a * nu_k + b * nu_kp
        sigma = dt_k

        fc, Ac, Bc    = lin_dyn(tau, x, u, params)

        P1_dot = (sigma * fc)
        P2_dot = (sigma * Ac @ phi_a).reshape((nz*nz,))
        P3_dot = (sigma * Ac @ phi_b_m + sigma * Bc * a).reshape((nz*n_nu,))
        P4_dot = (sigma * Ac @ phi_b_p + sigma * Bc * b).reshape((nz*n_nu,))
        P5_dot = (sigma * Ac @ phi_s   + fc)

        return jnp.concatenate([P1_dot, P2_dot, P3_dot, P4_dot, P5_dot])

    # rk4 single step function for jax integration
    def rk4_step_jax_bwd(tau, lds, nu_k, nu_kp, dt_k, params_jax):
        
        k1 = pack_lds_dot(tau, lds, nu_k, nu_kp, dt_k, params_jax)
        k2 = pack_lds_dot(tau + dt_sub / 2, lds + (dt_sub / 2) * k1, nu_k, nu_kp, dt_k, params_jax)
        k3 = pack_lds_dot(tau + dt_sub / 2, lds + (dt_sub / 2) * k2, nu_k, nu_kp, dt_k, params_jax)
        k4 = pack_lds_dot(tau + dt_sub, lds + dt_sub * k3, nu_k, nu_kp, dt_k, params_jax)
        
        lds_next = lds + (dt_sub / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return lds_next, None

    rk4_step_jax_jit_bwd = jax.jit(rk4_step_jax_bwd)

    # initilize stacked propagation vector  
    def pack_lds0(z_k):
        P1 = z_k
        P2 = jnp.eye(nz).reshape(nz*nz)
        P3 = jnp.zeros(nz*n_nu)
        P4 = jnp.zeros(nz*n_nu)
        P5 = jnp.zeros(nz)

        return jnp.concatenate([P1, P2, P3, P4, P5])

    # unpacks stacked propagation vector to correct shapes
    def unpack_ldsf(ldsf_k):
        z_minus_k = ldsf_k[ : Ak_ind0]
        A_jax_k    = ldsf_k[Ak_ind0  : Bk_ind0].reshape((nz, nz))
        B_jax_k    = ldsf_k[Bk_ind0  : Bkp_ind0].reshape((nz, n_nu))
        Bp_jax_k   = ldsf_k[Bkp_ind0 : Sk_ind0].reshape((nz, n_nu))
        S_jax_k    = ldsf_k[Sk_ind0  : ]

        return (A_jax_k, B_jax_k, Bp_jax_k, S_jax_k, z_minus_k)

    # propagation function for node k
    def propagate_k_bwd(k, z_ref, nu_ref, dt_ref, params_jax):

        z_kp  = z_ref[k+1]
        nu_k  = nu_ref[k]
        nu_kp = nu_ref[k+1]
        dt_k = dt_ref[k]

        # stack z, A, B, Bp, S for multiple shooting propagation
        lds0_k = pack_lds0(z_kp)

        # propagate stacked vector
        def rk4_step_jax_partial(lds, tau):
            return rk4_step_jax_jit_bwd(tau, lds, nu_k, nu_kp, dt_k, params_jax)

        ldsf_k, _ = jax.lax.scan(rk4_step_jax_partial, lds0_k, jnp.flip(t)[:-1])

        # unpack the stacked vector back into appropriate shapes
        return unpack_ldsf(ldsf_k)
    
    propagate_bwd = jax.jit(jax.vmap(propagate_k_bwd, in_axes=(0, None, None, None, None)))

    method.propagate_discretization_jax_bwd = propagate_bwd

# inverse free discretize with jax
def discretize_inv_free_jax(z_ref_np, nu_ref_np, dt_ref_np, problem, method):

    # convert numpy arrays to jax
    z_ref = jnp.asarray(z_ref_np)
    nu_ref = jnp.asarray(nu_ref_np)
    dt_ref = jnp.asarray(dt_ref_np)

    params = problem.params
    params_jax = tools.recursive_to_dict(params)

    # call jitted propagator for each node
    ks = jnp.arange(method.index_map.N['N'] - 1)
    A_jax, B_jax, Bp_jax, S_jax, z_minus = method.propagate_discretization_jax(ks, z_ref, nu_ref, dt_ref, params_jax)

    z_ref_0 = z_ref[[0], :]
    
    return np.asarray(A_jax), np.asarray(B_jax), np.asarray(Bp_jax), np.asarray(S_jax), np.asarray(jnp.vstack([z_ref_0, z_minus]))

# inverse free discretize with jax
def discretize_inv_free_jax_bwd(z_ref_np, nu_ref_np, dt_ref_np, problem, method):

    # convert numpy arrays to jax
    z_ref = jnp.asarray(z_ref_np)
    nu_ref = jnp.asarray(nu_ref_np)
    dt_ref = jnp.asarray(dt_ref_np)

    params = problem.params
    params_jax = tools.recursive_to_dict(params)

    # call jitted propagator for each node
    ks = jnp.arange(method.index_map.N['N'] - 1)
    A_jax, B_jax, Bp_jax, S_jax, z_minus = method.propagate_discretization_jax_bwd(ks, z_ref, nu_ref, dt_ref, params_jax)
    
    z_ref_f = z_ref[[-1], :]
    
    return np.asarray(A_jax), np.asarray(B_jax), np.asarray(Bp_jax), np.asarray(S_jax), np.asarray(jnp.vstack([z_minus, z_ref_f]))

def compute_linsys_discrete(z_ref, nu_ref, dt_ref, problem, method):
    """
    Compute the linear system in discrete form.

    Parameters:
    z_ref (numpy.ndarray): Reference state trajectory.
    nu_ref (numpy.ndarray): Reference control trajectory.
    dt_ref (numpy.ndarray): Time steps.
    trajopt_obj (dict): Dictionary containing trajopt_obj parameters.

    Returns:
    tuple: Ak, Bk, Bkp, Sk, z_minus
    """

    if method.flags["jax_dyn"]:
        Ak, Bk, Bkp, Sk, z_minus = discretize_inv_free_jax(z_ref, nu_ref, dt_ref, problem, method)
    else:
        if method.flags["ctcs"] != "none":
            Ak, Bk, Bkp, Sk, z_minus = discretize_ctcs(z_ref, nu_ref, dt_ref, problem, method)
        else:
            Ak, Bk, Bkp, Sk, z_minus = discretize_inv_foh(z_ref, nu_ref, dt_ref, problem, method)
    
    return Ak, Bk, Bkp, Sk, z_minus

def compute_linsys_discrete_bwd(z_ref, nu_ref, dt_ref, problem, method):

    Ak, Bk, Bkp, Sk, z_minus = discretize_inv_free_jax_bwd(z_ref, nu_ref, dt_ref, problem, method)

    return Ak, Bk, Bkp, Sk, z_minus 

# other discretization methods

# Compute exact discretize for linear dynamic system
def discretize_inv_foh(z_ref, nu_ref, dt_ref, problem, method):
    N = method.index_map.N['N']

    traj_minus_data = {"z_minus": [z_ref[0]]}

    # Precompute
    eye_flat = np.eye(problem.index_map.n['state']).ravel()

    # Stack dynamics
    lds0_stack = []
    for k in range(N - 1):
        method.lds0[method.z_ind] = z_ref[k]
        method.lds0[method.Ak_ind] = eye_flat
        lds0_stack.append(method.lds0.copy())

    lds0_stack = np.concatenate(lds0_stack)

    def derivs_step(tau, lds):
        return RHS_ltv(tau, lds, nu_ref, dt_ref, problem, method)

    sol = solve_ivp(derivs_step, [0, 1], lds0_stack, method="RK45", atol=1e-6, rtol=1e-6)

    lds_end = sol.y[:, -1]  # shape: (total_state_size,)

    assert lds_end.shape[0] == (N - 1) * method.lds0_size

    Ak  = np.zeros((N - 1, problem.index_map.n['state'], problem.index_map.n['state']))
    Bk  = np.zeros((N - 1, problem.index_map.n['state'], problem.index_map.n['control']))
    Bkp = np.zeros((N - 1, problem.index_map.n['state'], problem.index_map.n['control']))
    Sk  = np.zeros((N - 1, problem.index_map.n['state']))

    for k in range(N - 1):
        base    = k * method.lds0_size
        traj_minus_data["z_minus"].append(lds_end[base + method.z_ind])

        Ak_bar  = lds_end[base + method.Ak_ind].reshape(problem.index_map.n['state'], problem.index_map.n['state'])
        Bk_bar  = lds_end[base + method.Bk_ind].reshape(problem.index_map.n['state'], problem.index_map.n['control'])
        Bkp_bar = lds_end[base + method.Bkp_ind].reshape(problem.index_map.n['state'], problem.index_map.n['control'])
        Sk_bar  = lds_end[base + method.Sk_ind].reshape(problem.index_map.n['state'])

        Ak[k]   = Ak_bar
        Bk[k]   = Ak_bar @ Bk_bar
        Bkp[k]  = Ak_bar @ Bkp_bar
        Sk[k]   = Ak_bar @ Sk_bar

    z_minus    = np.array(traj_minus_data["z_minus"])

    return Ak, Bk, Bkp, Sk, z_minus

# Integrate linear system
def RHS_ltv(tau, lds, nu_ref, dt_ref, problem, method):



    # Initialize
    lds_dot         = np.zeros_like(lds)
    N               = method.index_map.N['N']

    # Extract times and FOH control input
    Om_k            = 1 - tau
    Om_kp           = tau

    nrows, ncols    = nu_ref.shape
    rows            = nrows if nrows > ncols else ncols

    v_1             = Om_k * np.ones((rows, 1))
    v_2             = Om_kp * np.ones((rows - 1, 1))
    Om              = np.diagflat(v_1) + np.diagflat(v_2, 1)

    u               = Om @ nu_ref

    for k in range(N - 1):
        dt_k       = dt_ref[k]

        # Extract state info
        x           = lds[ k * method.lds0_size + method.z_ind ]

        # Extract continuous time Jacobians
        fc, Ac, Bc = problem.lin_dyn(tau, x, u[k])

        # Extract STM
        Phi_tau     = lds[ k * method.lds0_size + method.Ak_ind ].reshape(model.n, model.n)

        # Construct Jacobians w.r.t. tau
        f_tau       = dt_k * fc
        A_tau       = dt_k * Ac
        B_tau       = dt_k * Om_k * Bc
        Bp_tau      = dt_k * Om_kp * Bc
        S_tau       = fc

        Phi_tau_inv = np.linalg.pinv(Phi_tau)

        # Construct derivatives
        x_dot       = f_tau
        A_tau_dot   = A_tau @ Phi_tau
        B_tau_dot   = Phi_tau_inv @ B_tau
        Bp_tau_dot  = Phi_tau_inv @ Bp_tau
        S_tau_dot   = Phi_tau_inv @ S_tau

        # Setup linear system properly
        lds_dot[ k * method.lds0_size + method.z_ind  ] = x_dot
        lds_dot[ k * method.lds0_size + method.Ak_ind ] = A_tau_dot.flatten()
        lds_dot[ k * method.lds0_size + method.Bk_ind ] = B_tau_dot.flatten()
        lds_dot[ k * method.lds0_size + method.Bkp_ind] = Bp_tau_dot.flatten()
        lds_dot[ k * method.lds0_size + method.Sk_ind ] = S_tau_dot

    return lds_dot


def discretize_ctcs(z_ref, nu_ref, dt_ref, problem, method):
    """
    Compute exact discretize for linear dynamic system.

    Parameters:
    z_ref (numpy.ndarray): Reference state trajectory.
    nu_ref (numpy.ndarray): Reference control trajectory.
    dt_ref (numpy.ndarray): Time steps.
    trajopt_obj (dict): Dictionary containing trajopt_obj parameters.

    Returns:
    tuple: Ak, Bk, Bkp, Sk, z_minus
    """

    N = method.index_map.N['N']

    traj_minus_data = {"z_minus": [z_ref[0]]}

    # Setup LTV system dynamics
    lds0_stack = []
    for k in range(N - 1):
        method.lds0[method.z_ind] = z_ref[k]
        method.lds0[method.Ak_ind] = np.reshape(np.eye(problem.index_map.n['z']), -1)
        lds0_stack.append(method.lds0.copy())

    lds0_stack = np.hstack(lds0_stack)

    def derivs_step(tau, lds):
        return RHS_ltv_ctcs(tau, lds, nu_ref, dt_ref, problem, method)

    sol = solve_ivp(derivs_step, [0, 1], lds0_stack, atol=1E-12, rtol=1E-12)
    lds_out_stack = sol.y

    Ak = np.zeros((N-1, problem.index_map.n['z'], problem.index_map.n['z']))
    Bk = np.zeros((N-1, problem.index_map.n['z'], problem.index_map.n['control']))
    Bkp = np.zeros((N-1, problem.index_map.n['z'], problem.index_map.n['control']))
    Sk = np.zeros((N-1, problem.index_map.n['z']))

    # Extract dense values
    for k in range(N - 1):
        lds_end = lds_out_stack[:, -1]
        traj_minus_data["z_minus"].append(
            lds_end[k * method.lds0_size + method.z_ind])

        # Reshape matrices
        Ak_bar = np.reshape(lds_end[k * method.lds0_size + method.Ak_ind],
                            (problem.index_map.n['z'], problem.index_map.n['z']))
        Bk_bar = np.reshape(lds_end[k * method.lds0_size + method.Bk_ind],
                            (problem.index_map.n['z'], problem.index_map.n['control']))
        Bkp_bar = np.reshape(lds_end[k * method.lds0_size + method.Bkp_ind],
                             (problem.index_map.n['z'], problem.index_map.n['control']))
        Sk_bar = lds_end[k * method.lds0_size + method.Sk_ind]

        # Fill in the next STM
        Ak[k]   = Ak_bar
        Bk[k]   = Ak_bar @ Bk_bar
        Bkp[k]  = Ak_bar @ Bkp_bar
        Sk[k]   = Ak_bar @ Sk_bar

    # Extract x_ref_minus traj (from integration)
    z_minus    = np.array(traj_minus_data["z_minus"])

    return Ak, Bk, Bkp, Sk, z_minus


def RHS_ltv_ctcs(tau, lds, nu_ref, dt_ref, problem, method):
    """
    Integrate linear system.

    Parameters:
    tau (float): Current time step.
    lds (numpy.ndarray): Linear dynamic system state.
    nu_ref (numpy.ndarray): Reference control trajectory.
    dt_ref (numpy.ndarray): Time steps.
    trajopt_obj (dict): Dictionary containing trajopt_obj parameters.

    Returns:
    numpy.ndarray: Derivative of the linear dynamic system state.
    """

    lds_dot = np.zeros_like(lds)
    N = method.index_map.N['N']

    Om_k = 1 - tau
    Om_kp = tau

    Om = np.diag(Om_k * np.ones(nu_ref.shape[0])) + np.diag(Om_kp * np.ones(nu_ref.shape[0] - 1), 1)

    u = Om @ nu_ref

    for k in range(N - 1):
        dt_k = dt_ref[k]
        x = lds[k * method.lds0_size + method.z_ind]

        Ac, Bc, fc = convexify.compute_ctcs_jacobians(tau, x, u[k, :], problem, method)

        Phi_tau = np.reshape(lds[k * method.lds0_size + method.Ak_ind],
                             (problem.index_map.n['z'], problem.index_map.n['z']))

        f_tau = dt_k * fc
        A_tau = dt_k * Ac
        B_tau = dt_k * Om_k * Bc
        Bp_tau = dt_k * Om_kp * Bc
        S_tau = fc

        Phi_tau_inv = np.linalg.inv(Phi_tau)

        x_dot = f_tau
        A_tau_dot = A_tau @ Phi_tau
        B_tau_dot = Phi_tau_inv @ B_tau
        Bp_tau_dot = Phi_tau_inv @ Bp_tau
        S_tau_dot = Phi_tau_inv @ S_tau

        lds_dot[k * method.lds0_size + method.z_ind] = x_dot
        lds_dot[k * method.lds0_size + method.Ak_ind] = np.reshape(A_tau_dot, -1)
        lds_dot[k * method.lds0_size + method.Bk_ind] = np.reshape(B_tau_dot, -1)
        lds_dot[k * method.lds0_size + method.Bkp_ind] = np.reshape(Bp_tau_dot, -1)
        lds_dot[k * method.lds0_size + method.Sk_ind] = S_tau_dot

    return lds_dot