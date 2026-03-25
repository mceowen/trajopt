import numpy as np
import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from scipy.integrate import solve_ivp
import trajopt.library.methods.convexify as convexify
import trajopt.utils.tools as tools
import time
import diffrax

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

def set_ltv_indices(problem, method):
    """
    Function to set Linear Time Varying (LTV) indices and initialize arrays.

    Parameters:
    params (dict): Dictionary containing parameters.

    Returns:
    dict: Updated params with LTV indices and initialized arrays.
    """

    nz              = problem.index_map.n['z']
    n_nu            = problem.index_map.n['control']
    N               = method.index_map.N['N']

    method.z_ind    = np.arange(0, nz)
    method.Ak_ind   = np.arange(method.z_ind[-1] + 1, method.z_ind[-1] + 1 + nz**2 )
    method.Bk_ind   = np.arange(method.Ak_ind[-1] + 1, method.Ak_ind[-1] + 1 + nz*n_nu )
    method.Bkp_ind  = np.arange(method.Bk_ind[-1] + 1, method.Bk_ind[-1] + 1 + nz*n_nu )
    method.Sk_ind   = np.arange(method.Bkp_ind[-1] + 1, method.Bkp_ind[-1] + 1 + nz )

    method.Ak       = np.zeros((method.index_map.N['N'] - 1, nz, nz))
    method.Bk       = np.zeros((method.index_map.N['N'] - 1, nz, n_nu))
    method.Bkp      = np.zeros((method.index_map.N['N'] - 1, nz, n_nu))
    method.Sk       = np.zeros((method.index_map.N['N'] - 1, nz, 1))

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
        
    t_jax  = jnp.asarray(t_ref)
    z_jax  = jnp.asarray(z_ref)[:, :n_x]
    nu_jax = jnp.asarray(u_ref)

    params_jax = tools.recursive_to_dict(problem.params)
    
    g     = np.zeros((N, n_ineq))
    dgdz  = np.zeros((N, n_ineq, n_x))
    dgdnu = np.zeros((N, n_ineq, n_nu))
    
    col_start = 0
    for constraint in problem.constraints.get(ct=0, type="nonconvex_inequality"):
        col_end = col_start + constraint.dimension
        nodes = np.asarray(constraint.nodes)

        t_nodes  = t_jax[nodes]
        z_nodes  = z_jax[nodes]
        nu_nodes = nu_jax[nodes]

        f, dfcn_dz, dfcn_du = constraint.g_aff_batched(t_nodes, z_nodes, nu_nodes, params_jax)

        g[nodes, col_start:col_end]        = np.asarray(f)
        dgdz[nodes, col_start:col_end, :]  = np.asarray(dfcn_dz)
        dgdnu[nodes, col_start:col_end, :] = np.asarray(dfcn_du)

        col_start = col_end
    
    return g, dgdz, dgdnu

def compute_nonconvex_costs(t_ref, z_ref, u_ref, problem, method):

    n_x  = problem.index_map.n['state']
    n_nu = problem.index_map.n['control']
    N    = method.index_map.N['N']
    
    t_jax  = jnp.asarray(t_ref)
    z_jax  = jnp.asarray(z_ref)
    nu_jax = jnp.asarray(u_ref)

    params_jax = tools.recursive_to_dict(problem.params)
    
    cost     = np.zeros((N, 1))
    dcostdz  = np.zeros((N, 1, n_x))
    dcostdnu = np.zeros((N, 1, n_nu))

    path_nodes = jnp.arange(N - 1)
    for cost_object in problem.costs.get(ct=0, type="nonconvex"):
        f, dfcn_dz, dfcn_du = cost_object.g_aff_batched(
            t_jax[path_nodes], z_jax[path_nodes], nu_jax[path_nodes], params_jax
        )
        cost[:N-1, 0]      += np.asarray(f).flatten()
        dcostdz[:N-1, 0, :] += np.asarray(dfcn_dz).reshape(N - 1, n_x)
        dcostdnu[:N-1, 0, :] += np.asarray(dfcn_du).reshape(N - 1, n_nu)
            
    for cost_object in problem.costs.get(type="nonconvex_terminal"):
        f, dfcn_dz, dfcn_du = cost_object.g_aff(
            t_jax[-1], z_jax[-1], nu_jax[-1], params_jax
        )
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
    nsub_nodes = 20
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
    
    def f_dot(tau, lds_k, args):
        nu_k, nu_kp, dt_k, params_jax = args
        return pack_lds_dot(tau, lds_k, nu_k, nu_kp, dt_k, params_jax)


    # # rk4 single step function for jax integration
    # def rk4_step_jax(tau, lds, nu_k, nu_kp, dt_k, params_jax):
        
    #     k1 = pack_lds_dot(tau, lds, nu_k, nu_kp, dt_k, params_jax)
    #     k2 = pack_lds_dot(tau + dt_sub / 2, lds + (dt_sub / 2) * k1, nu_k, nu_kp, dt_k, params_jax)
    #     k3 = pack_lds_dot(tau + dt_sub / 2, lds + (dt_sub / 2) * k2, nu_k, nu_kp, dt_k, params_jax)
    #     k4 = pack_lds_dot(tau + dt_sub, lds + dt_sub * k3, nu_k, nu_kp, dt_k, params_jax)
        
    #     lds_next = lds + (dt_sub / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    #     return lds_next, None

    # rk4_step_jax_jit = jax.jit(rk4_step_jax)



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

        # # propagate stacked vector
        # def rk4_step_jax_partial(lds, tau):
        #     return rk4_step_jax_jit(tau, lds, nu_k, nu_kp, dt_k, params_jax)

        # ldsf_k, _ = jax.lax.scan(rk4_step_jax_partial, lds0_k, t[:-1])
        
        term = diffrax.ODETerm(f_dot)

        solver = diffrax.Dopri5()
        stepsize_controller = diffrax.PIDController(rtol=1e-4, atol=1e-4)
        sol = diffrax.diffeqsolve(
            term,
            solver,
            0.0,
            1.0,
            0.05,
            lds0_k,
            stepsize_controller=stepsize_controller,
            args=(nu_k, nu_kp, dt_k, params_jax)
        )

        ldsf_k = sol.ys[-1]

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

    rk4_step_jax_jit_bwd = rk4_step_jax_bwd

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

    start = time.time()
    ks = jnp.arange(method.index_map.N['N'] - 1)
    A_jax, B_jax, Bp_jax, S_jax, z_minus = method.propagate_discretization_jax(ks, z_ref, nu_ref, dt_ref, params_jax)

    print(f"actual_prop_time: {(start - time.time())*1000}")
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


#####################################################
##### PSEUDOSPECTRAL AND POLYNOMIAL PROTOTYPES: #####
#####################################################
USE_SPARTAN = True

# ---------------------------------------------------------------------
# Legendre polynomial helper
# ---------------------------------------------------------------------
def compute_legendre(N: int, x: np.ndarray, use_spartan: bool = USE_SPARTAN) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the Legendre polynomial P_n(x) and its derivative P_n'(x).

    Inputs:
    N :             Polynomial order.
    x : Evaluation points.
    use_spartan :   Use SPARTAN-style Legendre recursive computation if True, scipy computation if False.

    Outputs:
    Pn :            Legendre polynomial evaluated at x.
    dPn :           Derivative d/dx P_n(x) evaluated at x.
    """
    x = np.asarray(x, dtype=float)

    if use_spartan:
        x = np.asarray(x, dtype=float)

        if N == 0:
            return np.ones_like(x), np.zeros_like(x)
        elif N == 1:
            return x.copy(), np.ones_like(x)

        # P_{0}, P'_{0}
        Pn1 = np.ones_like(x)
        Dn1 = np.zeros_like(x)

        # P_{1}, P'_{1}
        Pn = x.copy()
        Dn = np.ones_like(x)

        for jj in range(2, N + 1):
            k = jj - 1  # current recurrence index, building P_{k+1}

            # Standard Legendre recurrence:
            # P_{k+1} = ((2k+1)x P_k - k P_{k-1}) / (k+1)
            P_temp = ((2 * k + 1) * x * Pn - k * Pn1) / (k + 1)

            # Derivative of the standard Legendre recurrence:
            # P'_{k+1} = ((2k+1)(P_k + x P'_k) - k P'_{k-1}) / (k+1)
            D_temp = ((2 * k + 1) * Pn + (2 * k + 1) * x * Dn - k * Dn1) / (k + 1)

            Pn1, Dn1 = Pn, Dn
            Pn, Dn = P_temp, D_temp

        return Pn, Dn
    
    else:
        if N == 0:
            return np.ones_like(x), np.zeros_like(x)

        Pn      = sp.special.eval_legendre(N, x)
        Pnm1    = sp.special.eval_legendre(N - 1, x)

        dPn     = N * (x * Pn - Pnm1) / (x**2 - 1.0)

        return Pn, dPn
 
# ---------------------------------------------------------------------
# Flipped Legendre-Radau polynomial
# ---------------------------------------------------------------------
def flipped_radau_polynomial(N: int, tau: np.ndarray, use_spartan: bool = USE_SPARTAN) -> np.ndarray:
    """
    Inputs:
    N :             Polynomial order
    tau :           Points to evaluate.
    use_spartan :   Legendre polynomimal computation method

    Outputs:        R_n(tau).
    """
    tau     = np.asarray(tau, dtype=float)

    Ln, _   = compute_legendre(N, tau, use_spartan=use_spartan)
    Lnm1, _ = compute_legendre(N - 1, tau, use_spartan=use_spartan)

    return Ln - Lnm1


def flipped_radau_polynomial_derivative(N: int, tau: np.ndarray) -> np.ndarray:
    """
    Derivative of the flipped Legendre-Radau polynomial
    """
    tau         = np.asarray(tau, dtype=float)
    _, dLn      = compute_legendre(N, tau, use_spartan=USE_SPARTAN)
    _, dLnm1    = compute_legendre(N - 1, tau, use_spartan=USE_SPARTAN)
    
    return dLn - dLnm1


# ---------------------------------------------------------------------
# Compute flipped Radau nodes and quadrature weights
# ---------------------------------------------------------------------
# computes flipped Radau collocation nodes, full node set, quadrature weights
def flipped_radau_nodes_and_weights(N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inputs:
    N :     # of collocation nodes.

    Outputs:
    tau :   Collocation nodes (including +1, excluding -1).
    etau :  Full discrete node set for interpolation (includes -1).
    w :     Quadrature weights associated with tau-vector.
    """
    if N < 1:
        raise ValueError("N must be >= 1")

    # Degenerate one-node case.
    if N == 1:
        tau     = np.array([1.0])
        etau    = np.array([-1.0, 1.0])
        w       = np.array([2.0])
        return tau, etau, w

    # -------------------------------------------------------------
    # Compute LGR nodes with Newton-Raphson, flip to get fLGR.
    # -------------------------------------------------------------
    tau_std = -np.cos(2.0 * np.pi * np.arange(N) / (2 * (N - 1) + 1))
    tau_std = tau_std.astype(float)

    def radau_polynomial(x):
        Ln, _   = compute_legendre(N, np.array([x]), use_spartan=USE_SPARTAN)
        Lnm1, _ = compute_legendre(N - 1, np.array([x]), use_spartan=USE_SPARTAN)
        return float(Ln[0] + Lnm1[0])


    def radau_polynomial_derivative(x):
        _, dLn  = compute_legendre(N, np.array([x]), use_spartan=USE_SPARTAN)
        _, dLnm1= compute_legendre(N - 1, np.array([x]), use_spartan=USE_SPARTAN)
        return float(dLn[0] + dLnm1[0])

    # SPARTAN-style hardcoded Newton-Raphson iteration for LGR nodes
    if USE_HARDCODED_NEWTON:

        tau_old = np.ones_like(tau_std) * 2.0
        eps_tol = np.finfo(float).eps

        L       = np.zeros((N, N+1))
        idx     = np.arange(1, N)

        while np.max(np.abs(tau_std - tau_old)) > eps_tol:

            tau_old = tau_std.copy()

            # Construct Legendre Vandermonde matrix
            L[0, :] = (-1) ** np.arange(N+1)

            L[idx, 0] = 1
            L[idx, 1] = tau_std[idx]

            for k in range(2, N+1):
                L[idx, k] = ((2*k-1) * tau_std[idx] * L[idx, k-1] - (k-1) * L[idx, k-2]) / k

            tau_std[idx] = tau_old[idx] - ((1 - tau_old[idx]) / N) * \
                (L[idx, N-1] + L[idx, N]) / (L[idx, N-1] - L[idx, N])
    
    # SciPy-based Newton-Raphson root finding for LGR nodes
    else:

        for j in range(1, N):
            x0 = tau_std[j]
            tau_std[j] = sp.optimize.newton(
                radau_polynomial,
                x0,
                fprime=radau_polynomial_derivative,
                tol=1e-14,
                maxiter=100,
            )

    # extrapolate tau_std to get the full node set, then flip for fLGR
    tau     = np.sort(-tau_std)
    etau    = np.concatenate(([-1.0], tau))

    # -------------------------------------------------------------
    # Compute quadrature weights.
    # -------------------------------------------------------------
    weights_std = np.zeros(N)
    weights_std[0] = 2.0 / N**2
    for j in range(1, N):
        Ln, _ = compute_legendre(N - 1, np.array([tau_std[j]]), use_spartan=USE_SPARTAN)
        weights_std[j] = (1.0 - tau_std[j]) / (N * Ln[0]) ** 2

    w = np.flip(weights_std)

    return tau, etau, w


# ---------------------------------------------------------------------
# Differentiation matrix
# ---------------------full_nodes------------------------------------------------
# computes the differentiation matrix D, shape (N, N+1)

def differentiation_matrix(etau: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inputs:  - 1D array of length m = N+1 containing the full node set (including -1).
    Outputs: D - differentiation matrix of shape (N, N+1) mapping state values at `full_nodes`
                 to derivatives at the collocation nodes (all nodes except the initial node).
    """
    xxPlusEnd = np.asarray(etau, dtype=float)
    M = len(xxPlusEnd)
    # M1 = M + 1
    # M2 = M * M

    # compute the barycentric weights
    Y       = np.tile(xxPlusEnd.reshape(-1, 1), (1, M))
    Ydiff   =  Y - Y.T + np.eye(M)

    WW      = np.tile((1.0 / np.prod(Ydiff, axis=1)).reshape(-1, 1), (1, M))
    D       = WW / (WW.T * Ydiff)

    # MATLAB: D(1:M1:M2) = 1-sum(D);
    np.fill_diagonal(D, 1.0 - np.sum(D, axis=0))

    # full differentiation matrix
    D       = -D.T
    D_full  = D.copy()

    # fLGR D-matrix
    D       = D[1:M, :]

    D2      = D @ D_full

    return D, D_full, D2

def differentiation_matrix_compare(full_nodes: np.ndarray) -> np.ndarray:
    x = np.asarray(full_nodes, dtype=float)
    m = len(x)

    # Pairwise differences: dX[i,j] = x_i - x_j
    dX = x[:, None] - x[None, :]

    # Barycentric weights:
    #   lambda_i = 1 / prod_{j != i} (x_i - x_j)
    dX_no_diag = dX + np.eye(m)
    lam = 1.0 / np.prod(dX_no_diag, axis=1)

    # Off-diagonal entries:
    #   D_ij = lambda_j / (lambda_i * (x_i - x_j)),  i != j
    D_full = np.outer(1.0 / lam, lam) / dX_no_diag

    # Fix diagonal entries so each row sums to zero
    np.fill_diagonal(D_full, 0.0)
    np.fill_diagonal(D_full, -np.sum(D_full, axis=1))

    # fLGR: remove the first row, keep all columns
    D = D_full[1:, :]

    return D


def flipped_radau_differential_operator(N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Inputs:
    N : Number of collocation nodes.

    Outputs:
    tau : Collocation nodes.
    etau : Full node set including -1.
    w : Quadrature weights.
    D : Differential operator mapping state values at etau to derivatives at collocation nodes tau.
    """
    tau, etau, w    = flipped_radau_nodes_and_weights(N)
    D, _, _    = differentiation_matrix(etau)

    #D_compare       = differentiation_matrix_compare(etau)
    #D = D_compare
   
    return tau, etau, w, D


# ---------------------------------------------------------------------
# Lagrange interpolation (Eq. 9, CEAS2017)
# ---------------------------------------------------------------------

def lagrange_basis(eval_points, nodes):
    """
    Compute Lagrange basis polynomials P_i(t) evaluated at eval_points.
    """
    t = np.asarray(eval_points, dtype=float)
    ti = np.asarray(nodes, dtype=float)

    m_eval = len(t)
    n_nodes = len(ti)

    P = np.ones((m_eval, n_nodes))

    for i in range(n_nodes):
        for k in range(n_nodes):
            if k != i:
                P[:, i] *= (t - ti[k]) / (ti[i] - ti[k])

    return P


def lagrange_interpolate(eval_points, nodes, values):
    """
    Evaluate interpolating polynomial.
    """
    P = lagrange_basis(eval_points, nodes)
    return P @ values

