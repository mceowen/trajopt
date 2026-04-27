import numpy as np
import jax
import jax.numpy as jnp
import cvxpy as cp

jax.config.update("jax_enable_x64", True)
import diffrax


import trajopt.methods.scp.pseudospectral as pseudospectral
import trajopt.utils.tools as tools

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

    nz              = problem.index_map.n.z
    n_nu            = problem.index_map.n.nu
    N               = method.index_map.N.time_grid

    method.z_ind    = np.arange(0, nz)
    method.Ak_ind   = np.arange(method.z_ind[-1] + 1, method.z_ind[-1] + 1 + nz**2 )
    method.Bk_ind   = np.arange(method.Ak_ind[-1] + 1, method.Ak_ind[-1] + 1 + nz*n_nu )
    method.Bkp_ind  = np.arange(method.Bk_ind[-1] + 1, method.Bk_ind[-1] + 1 + nz*n_nu )
    method.Ak       = np.zeros((method.index_map.N.time_grid - 1, nz, nz))
    method.Bk       = np.zeros((method.index_map.N.time_grid - 1, nz, n_nu))
    method.Bkp      = np.zeros((method.index_map.N.time_grid - 1, nz, n_nu))

    method.lds0_size= method.Bkp_ind[-1] + 1
    method.lds0     = np.zeros( method.lds0_size )

    method.lds0[method.Ak_ind] = np.reshape(np.eye(nz), -1)

    # convert indeces to jax arrays for jax discretize option
    method.lds0_size_jax = int(method.lds0_size)
    method.z_ind_jax     = jnp.asarray(method.z_ind)  
    method.Ak_ind_jax    = jnp.asarray(method.Ak_ind)
    method.Bk_ind_jax    = jnp.asarray(method.Bk_ind)
    method.Bkp_ind_jax   = jnp.asarray(method.Bkp_ind)

    

def compute_nonconvex_constraints(z, nu, problem, method):
    n_ineq    = problem.index_map.n.nonconvex_inequality
    n_z       = problem.index_map.n.z
    n_nu       = problem.index_map.n.nu
    N         = method.index_map.N.time_grid
    z_jax     = jnp.asarray(z)
    nu_jax    = jnp.asarray(nu)
    params_jax = problem.params

    g     = np.zeros((N, n_ineq))
    dgdz  = np.zeros((N, n_ineq, n_z))
    dgdnu = np.zeros((N, n_ineq, n_nu))

    col_start = 0
    for constraint in problem.constraints.get(ct=0, type="nonconvex_inequality"):
        col_end = col_start + constraint.dimension
        nodes = np.asarray(constraint.nodes)

        f_batch, dfdx_batch, dfdu_batch = constraint.g_aff_batched(z_jax[nodes], nu_jax[nodes], params_jax)

        g[nodes, col_start:col_end]      = np.asarray(f_batch)
        dgdz[nodes, col_start:col_end, :] = np.asarray(dfdx_batch)
        dgdnu[nodes, col_start:col_end, :] = np.asarray(dfdu_batch)

        col_start = col_end

    return g, dgdz, dgdnu


def compute_nonconvex_costs(z, nu, problem, method):
    N         = method.index_map.N.time_grid
    n_z       = problem.index_map.n.z
    n_nu       = problem.index_map.n.nu
    idx       = problem.index_map.indices

    cost     = np.zeros((N, 1))
    dcostdz  = np.zeros((N, 1, n_z))
    dcostdnu = np.zeros((N, 1, n_nu))

    params_jax = problem.params
    nonconvex_costs = problem.costs.get(type="nonconvex")

    if len(nonconvex_costs) == 0:
        return cost, dcostdz, dcostdnu

    z_jax  = jnp.asarray(z)
    nu_jax = jnp.asarray(nu)

    for cost_fn in nonconvex_costs:
        f_batch, dfdx_batch, dfdu_batch = cost_fn.g_aff_batched(z_jax, nu_jax, params_jax)

        f_np    = np.asarray(f_batch).reshape(N, -1)
        dfdx_np = np.asarray(dfdx_batch).reshape(N, -1, n_z)
        dfdu_np = np.asarray(dfdu_batch).reshape(N, -1, n_nu)

        cost[:, 0]                     += np.sum(f_np, axis=1)
        dcostdz[:, 0, :]              += np.sum(dfdx_np, axis=1)
        dcostdnu[:, 0, :] += np.sum(dfdu_np, axis=1)

    return cost, dcostdz, dcostdnu




def compile_jax_discretization(problem, method):
    n_z  = problem.index_map.n.z
    n_nu = problem.index_map.n.nu

    # define static indices for stacked RHS vector 
    Ak_ind0   = n_z
    Bk_ind0   = Ak_ind0  + n_z*n_z
    Bkp_ind0  = Bk_ind0  + n_z*n_nu

    z_time_idx = problem.index_map.indices.z.time

    params_jax = problem.params

    # pull ltv dynamics
    lin_dyn = problem.constraints.get(type='dynamics')[0].lin_dyn

    # packs the derivative of stacked RHS vector for node k
    def pack_lds_dot(tau, k, lds_k, nu_k, nu_kp, params_jax):

        z       = lds_k[         : Ak_ind0]
        phi_a   = lds_k[Ak_ind0  : Bk_ind0].reshape((n_z, n_z))
        phi_b_m = lds_k[Bk_ind0  : Bkp_ind0].reshape((n_z, n_nu))
        phi_b_p = lds_k[Bkp_ind0 : ].reshape((n_z, n_nu))

        tau_k = k / (problem.index_map.N.time_grid - 1)
        tau_kp = (k+1) / (problem.index_map.N.time_grid - 1)
        a = (tau_kp - tau) / (tau_kp - tau_k)
        b = (tau - tau_k) / (tau_kp - tau_k)
        nu = a * nu_k + b * nu_kp

        fc, Ac, Bc    = lin_dyn(z, nu, params_jax)

        P1_dot = fc
        P2_dot = (Ac @ phi_a).reshape((n_z*n_z,))
        P3_dot = (Ac @ phi_b_m + Bc * a).reshape((n_z*n_nu,))
        P4_dot = (Ac @ phi_b_p + Bc * b).reshape((n_z*n_nu,))

        return jnp.concatenate([P1_dot, P2_dot, P3_dot, P4_dot])

    # define dynamics function for diffrax integrator
    def f_dot(tau, lds_k, args):
        k, nu_k, nu_kp, params_jax = args
        return pack_lds_dot(tau, k, lds_k, nu_k, nu_kp, params_jax)

    # initilize stacked propagation vector  
    def pack_lds0(z_k):
        P1 = z_k
        P2 = jnp.eye(n_z).reshape(n_z*n_z)
        P3 = jnp.zeros(n_z*n_nu)
        P4 = jnp.zeros(n_z*n_nu)

        return jnp.concatenate([P1, P2, P3, P4])

    # unpacks stacked propagation vector to correct shapes
    def unpack_ldsf(ldsf_k):
        z_minus_k = ldsf_k[ : Ak_ind0]
        A_jax_k    = ldsf_k[Ak_ind0  : Bk_ind0].reshape((n_z, n_z))
        B_jax_k    = ldsf_k[Bk_ind0  : Bkp_ind0].reshape((n_z, n_nu))
        Bp_jax_k   = ldsf_k[Bkp_ind0 : ].reshape((n_z, n_nu))

        return (A_jax_k, B_jax_k, Bp_jax_k, z_minus_k)

    def _apply_phase_schedule(params_jax, k):
        return tools.AttrDict({key: (jnp.asarray(val)[k] if hasattr(val, '__len__') and not isinstance(val, dict) else val)
                               for key, val in params_jax.items()})

    use_fixed_dt = int(getattr(method.flags, 'ode_fixed_dt', 0))
    N_grid = problem.index_map.N.time_grid

    if use_fixed_dt:
        nsub = 20
        delta_tau = 1.0 / (N_grid - 1)
        dt_rk4 = delta_tau / nsub

        def propagate_k(k, z_ref, nu_ref, params_jax):
            params_k = _apply_phase_schedule(params_jax, k)

            z_k   = z_ref[k]
            nu_k  = nu_ref[k]
            nu_kp = nu_ref[k+1]
            lds0_k = pack_lds0(z_k)

            tau_k = k / (N_grid - 1)
            taus  = tau_k + jnp.arange(nsub) * dt_rk4

            def rk4_step(lds, tau):
                k1 = pack_lds_dot(tau,              k, lds,                     nu_k, nu_kp, params_k)
                k2 = pack_lds_dot(tau + dt_rk4/2,   k, lds + (dt_rk4/2) * k1,  nu_k, nu_kp, params_k)
                k3 = pack_lds_dot(tau + dt_rk4/2,   k, lds + (dt_rk4/2) * k2,  nu_k, nu_kp, params_k)
                k4 = pack_lds_dot(tau + dt_rk4,     k, lds + dt_rk4 * k3,      nu_k, nu_kp, params_k)
                return lds + (dt_rk4 / 6) * (k1 + 2*k2 + 2*k3 + k4), None

            ldsf_k, _ = jax.lax.scan(rk4_step, lds0_k, taus)
            return unpack_ldsf(ldsf_k)

    else:
        def propagate_k(k, z_ref, nu_ref, params_jax):
            params_k = _apply_phase_schedule(params_jax, k)

            z_k   = z_ref[k]
            nu_k  = nu_ref[k]
            nu_kp = nu_ref[k+1]
            lds0_k = pack_lds0(z_k)

            tau_k  = k / (N_grid - 1)
            tau_kp = (k+1) / (N_grid - 1)

            term = diffrax.ODETerm(f_dot)
            solver = diffrax.Dopri5()
            stepsize_controller = diffrax.PIDController(rtol=1e-7, atol=1e-7)
            sol = diffrax.diffeqsolve(
                term,
                solver,
                tau_k,
                tau_kp,
                0.00005,
                lds0_k,
                stepsize_controller=stepsize_controller,
                args=(k, nu_k, nu_kp, params_k),
                max_steps=65536,
            )
            ldsf_k = sol.ys[-1]
            return unpack_ldsf(ldsf_k)

    propagate = jax.jit(jax.vmap(propagate_k, in_axes=(0, None, None, None)))
    method.propagate_discretization_jax = propagate


def compile_jax_discretization_bwd(problem, method):
    nz = problem.index_map.n.z
    n_nu = problem.index_map.n.nu

    # define static indices for stacked RHS vector 
    Ak_ind0   = nz
    Bk_ind0   = Ak_ind0  + nz*nz
    Bkp_ind0  = Bk_ind0  + nz*n_nu

    z_time_idx = problem.index_map.indices.z.time

    params = problem.params

    # pull ltv dynamics
    lin_dyn = problem.constraints.get(type='dynamics')[0].lin_dyn

    # nsub defines the number of sub *nodes* between knot points
    nsub_nodes = 20
    dt_sub = -1.0 / (nsub_nodes + 1)
    t = jnp.linspace(0.0, 1.0, nsub_nodes + 2)

    # packs the derivative of stacked RHS vector for node k
    def pack_lds_dot(tau, lds_k, nu_k, nu_kp, params_jax):

        z       = lds_k[         : Ak_ind0]
        phi_a   = lds_k[Ak_ind0  : Bk_ind0].reshape((nz, nz))
        phi_b_m = lds_k[Bk_ind0  : Bkp_ind0].reshape((nz, n_nu))
        phi_b_p = lds_k[Bkp_ind0 : ].reshape((nz, n_nu))

        a = 1 - tau
        b = tau
        nu = a * nu_k + b * nu_kp

        # TODO(Skye): Modify lin_dyn to take in just z,nu 
        # TODO(Skye): Verify that fc,Ac,Bc correct for given timestep using
        t_node = z[z_time_idx][0] if len(z_time_idx) > 0 else tau
        fc, Ac, Bc    = lin_dyn(t_node, z, nu, params_jax)

        P1_dot = fc
        P2_dot = (Ac @ phi_a).reshape((nz*nz,))
        P3_dot = (Ac @ phi_b_m + Bc * a).reshape((nz*n_nu,))
        P4_dot = (Ac @ phi_b_p + Bc * b).reshape((nz*n_nu,))

        return jnp.concatenate([P1_dot, P2_dot, P3_dot, P4_dot])

    # rk4 single step function for jax integration
    def rk4_step_jax_bwd(tau, lds, nu_k, nu_kp, params_jax):
        # TOD(Skye): ensure that dt_sub is computed from the z vector correctly for each node
        k1 = pack_lds_dot(tau, lds, nu_k, nu_kp, params_jax)
        k2 = pack_lds_dot(tau + dt_sub / 2, lds + (dt_sub / 2) * k1, nu_k, nu_kp, params_jax)
        k3 = pack_lds_dot(tau + dt_sub / 2, lds + (dt_sub / 2) * k2, nu_k, nu_kp, params_jax)
        k4 = pack_lds_dot(tau + dt_sub, lds + dt_sub * k3, nu_k, nu_kp, params_jax)
        
        lds_next = lds + (dt_sub / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return lds_next, None

    rk4_step_jax_jit_bwd = rk4_step_jax_bwd

    # initilize stacked propagation vector  
    def pack_lds0(z_k):
        P1 = z_k
        P2 = jnp.eye(nz).reshape(nz*nz)
        P3 = jnp.zeros(nz*n_nu)
        P4 = jnp.zeros(nz*n_nu)

        return jnp.concatenate([P1, P2, P3, P4])

    # unpacks stacked propagation vector to correct shapes
    def unpack_ldsf(ldsf_k):
        z_minus_k = ldsf_k[ : Ak_ind0]
        A_jax_k    = ldsf_k[Ak_ind0  : Bk_ind0].reshape((nz, nz))
        B_jax_k    = ldsf_k[Bk_ind0  : Bkp_ind0].reshape((nz, n_nu))
        Bp_jax_k   = ldsf_k[Bkp_ind0 : ].reshape((nz, n_nu))

        return (A_jax_k, B_jax_k, Bp_jax_k, z_minus_k)

    # propagation function for node k
    def propagate_k_bwd(k, z_ref, nu_ref, params_jax):

        z_kp  = z_ref[k+1]
        nu_k  = nu_ref[k]
        nu_kp = nu_ref[k+1]
        # stack z, A, B, Bp for multiple shooting propagation
        lds0_k = pack_lds0(z_kp)

        # propagate stacked vector
        def rk4_step_jax_partial(lds, tau):
            return rk4_step_jax_jit_bwd(tau, lds, nu_k, nu_kp, params_jax)

        ldsf_k, _ = jax.lax.scan(rk4_step_jax_partial, lds0_k, jnp.flip(t)[:-1])

        # unpack the stacked vector back into appropriate shapes
        return unpack_ldsf(ldsf_k)
    
    propagate_bwd = jax.jit(jax.vmap(propagate_k_bwd, in_axes=(0, None, None, None)))

    method.propagate_discretization_jax_bwd = propagate_bwd




# inverse free discretize with jax
def discretize_inv_free_jax(z_ref_np, nu_ref_np, problem, method):

    # convert numpy arrays to jax
    z_ref       = jnp.asarray(z_ref_np)
    nu_ref      = jnp.asarray(nu_ref_np)
    params_jax = problem.params

    # TODO(Skye): ensure dtk is computed from the z vector correctly for each node in the jax propagator
    # call jitted propagator for each node
    ks = jnp.arange(method.index_map.N.time_grid - 1)
    A_jax, B_jax, Bp_jax, z_minus = method.propagate_discretization_jax(ks, z_ref, nu_ref, params_jax)

    # print(f"actual_prop_time: {(start - time.time())*1000}")
    z_ref_0 = z_ref[[0], :]
    
    return np.asarray(A_jax), np.asarray(B_jax), np.asarray(Bp_jax), np.asarray(jnp.vstack([z_ref_0, z_minus]))



# inverse free discretize with jax
def discretize_inv_free_jax_bwd(z_ref_np, nu_ref_np, problem, method):

    # convert numpy arrays to jax
    z_ref = jnp.asarray(z_ref_np)
    nu_ref = jnp.asarray(nu_ref_np)
    params_jax = problem.params

    # call jitted propagator for each node
    ks = jnp.arange(method.index_map.N.time_grid - 1)
    A_jax, B_jax, Bp_jax, z_minus = method.propagate_discretization_jax_bwd(ks, z_ref, nu_ref, params_jax)
    
    z_ref_f = z_ref[[-1], :]
    
    return np.asarray(A_jax), np.asarray(B_jax), np.asarray(Bp_jax), np.asarray(jnp.vstack([z_minus, z_ref_f]))




def compute_linsys_discrete(z_ref, nu_ref, problem, method):
    """
    Compute the linear system in discrete form.

    Parameters:
    z_ref (numpy.ndarray): Reference state trajectory.
    nu_ref (numpy.ndarray): Reference control trajectory.
    t_ref (numpy.ndarray): Node time values.
    trajopt_obj (dict): Dictionary containing trajopt_obj parameters.

    Returns:
    tuple: Ak, Bk, Bkp, z_minus
    """

    if method.flags.jax_dyn:
        Ak, Bk, Bkp, z_minus = discretize_inv_free_jax(z_ref, nu_ref, problem, method)
    else:
       pass # TODO(Skye/Carlos): add sympy and numpy(analytical) backend here

    return Ak, Bk, Bkp, z_minus



def compute_linsys_discrete_bwd(z_ref, nu_ref, problem, method):

    Ak, Bk, Bkp, z_minus = discretize_inv_free_jax_bwd(z_ref, nu_ref, problem, method)

    return Ak, Bk, Bkp, z_minus


def build_ms_dyn_constraint(subproblem, k):
    """
    Construct the multiple shooting dynamics constraint for time step k.

    The constraint enforces:
        dz[k+1] + z_ref[k+1] - z_m[k+1] == Ak[k] @ dz[k] + Bk[k] @ dnu[k] + Bkp[k] @ dnu[k+1] + (vb_dyn_p[k] - vb_dyn_m[k])

    Returns:
    --------
    cp.Constraint
        The equality constraint for the multiple shooting dynamics
    """
    rhs = (
        subproblem.Ak[k] @ subproblem.dz[k]
        + subproblem.Bk[k] @ subproblem.dnu[k]
        + subproblem.Bkp[k] @ subproblem.dnu[k + 1]
        + subproblem.vb_stack.dynamics[k]
    )
    return subproblem.dz[k + 1] + subproblem.z_ref[k + 1] - subproblem.z_m[k + 1] == rhs


def build_ps_dyn_constraints(subproblem):
    """
    Build pseudospectral dynamics constraints as a single block matrix equation.
    At collocation node k: state is z[k+1], control is nu[k+1] (both at etau[k+1]).

    Returns:
    --------
    cp.Constraint
        Single constraint for all collocation nodes, shape (N_col, n_z)
    """
    method = subproblem.method
    N_col  = method.index_map.N.time_grid - 1

    Z   = subproblem.z_ref + subproblem.dz
    lhs = 2.0 * (subproblem.ps_D @ Z)

    rhs_list = []
    for k in range(N_col):
        rhs_k = subproblem.ps_f_ref[k] + subproblem.ps_Ac[k] @ subproblem.dz[k + 1] + subproblem.ps_Bc[k] @ subproblem.dnu[k + 1]
        rhs_list.append(rhs_k)

    rhs = cp.vstack(rhs_list)

    return lhs == rhs


def compute_ps_dynamics_and_jacobians(z_ref, nu_ref, problem, method):
    """
    Compute dynamics and Jacobians at reference trajectory for pseudospectral collocation.
    State and control at collocation node k are both at etau[k+1]: z[k+1], nu[k+1].
    """
    N_col = problem.index_map.N.time_grid - 1
    n_z = problem.index_map.n.z
    n_nu = problem.index_map.n.nu

    lin_dyn = problem.constraints.get(type='dynamics')[0].lin_dyn
    params_jax = problem.params

    f_ref_col = np.zeros((N_col, n_z))
    Ac_col = np.zeros((N_col, n_z, n_z))
    Bc_col = np.zeros((N_col, n_z, n_nu))

    for k in range(N_col):
        z_k = np.asarray(z_ref[k + 1])
        nu_k = np.asarray(nu_ref[k + 1])

        fc_k, Ac_k, Bc_k = lin_dyn(z_k, nu_k, params_jax)

        f_ref_col[k, :] = np.asarray(fc_k)
        Ac_col[k, :, :] = np.asarray(Ac_k)
        Bc_col[k, :, :] = np.asarray(Bc_k)

    return f_ref_col, Ac_col, Bc_col


def compute_ps_differentiation_matrix(N_col):
    """
    Compute the pseudospectral differentiation matrix D for fLGR collocation.
    """
    tau, etau, w, D = pseudospectral.flipped_radau_differential_operator(N_col)
    return tau, etau, w, D



##############################################################################################################
# ORIGINAL DISCRETIZATION METHODS - DEPRECATED
##############################################################################################################

# TODO(Skye): Clean up and make work for numpy or sympy! Currently deprecated


# def _coerce_time_grid(t_ref, N):
#     t_arr = np.asarray(t_ref).reshape(-1)

#     # Transition helper:
#     # - preferred input is node times of length N
#     # - legacy support for interval lengths of length N-1
#     if t_arr.size == N - 1:
#         return np.concatenate(([0.0], np.cumsum(t_arr)))

#     if t_arr.size != N:
#         raise ValueError(f"Expected t_ref with length {N} (or legacy dt_ref with length {N-1}), got {t_arr.size}.")

#     return t_arr

# def compute_linsys_discrete(z_ref, nu_ref, t_ref, problem, method):
#     """
#     Compute the linear system in discrete form.

#     Parameters:
#     z_ref (numpy.ndarray): Reference state trajectory.
#     nu_ref (numpy.ndarray): Reference control trajectory.
#     t_ref (numpy.ndarray): Node time values.
#     trajopt_obj (dict): Dictionary containing trajopt_obj parameters.

#     Returns:
#     tuple: Ak, Bk, Bkp, z_minus
#     """

#     if method.flags.jax_dyn:
#         Ak, Bk, Bkp, z_minus = discretize_inv_free_jax(z_ref, nu_ref, problem, method)
#     else:
#         t_ref = _coerce_time_grid(t_ref, method.index_map.N.time_grid)
#         if method.flags.ctcs != "none":
#             Ak, Bk, Bkp, z_minus = discretize_ctcs(z_ref, nu_ref, t_ref, problem, method)
#         else:
#             Ak, Bk, Bkp, z_minus = discretize_inv_foh(z_ref, nu_ref, t_ref, problem, method)

#         return Ak, Bk, Bkp, z_minus

#     return Ak, Bk, Bkp, z_minus

# Compute exact discretize for linear dynamic system
# def discretize_inv_foh(z_ref, nu_ref, t_ref, problem, method):
#     N = method.index_map.N.time_grid

#     traj_minus_data = {"z_minus": [z_ref[0]]}

#     # Precompute
#     eye_flat = np.eye(problem.index_map.n.state).ravel()

#     # Stack dynamics
#     lds0_stack = []
#     for k in range(N - 1):
#         method.lds0[method.z_ind] = z_ref[k]
#         method.lds0[method.Ak_ind] = eye_flat
#         lds0_stack.append(method.lds0.copy())

#     lds0_stack = np.concatenate(lds0_stack)

#     def derivs_step(tau, lds):
#         return RHS_ltv(tau, lds, nu_ref, t_ref, problem, method)

#     sol = solve_ivp(derivs_step, [0, 1], lds0_stack, method="RK45", atol=1e-6, rtol=1e-6)

#     lds_end = sol.y[:, -1]  # shape: (total_state_size,)

#     assert lds_end.shape[0] == (N - 1) * method.lds0_size

#     Ak  = np.zeros((N - 1, problem.index_map.n.state, problem.index_map.n.state))
#     Bk  = np.zeros((N - 1, problem.index_map.n.state, problem.index_map.n.nu))
#     Bkp = np.zeros((N - 1, problem.index_map.n.state, problem.index_map.n.nu))

#     for k in range(N - 1):
#         base    = k * method.lds0_size
#         traj_minus_data["z_minus"].append(lds_end[base + method.z_ind])

#         Ak_bar  = lds_end[base + method.Ak_ind].reshape(problem.index_map.n.state, problem.index_map.n.state)
#         Bk_bar  = lds_end[base + method.Bk_ind].reshape(problem.index_map.n.state, problem.index_map.n.nu)
#         Bkp_bar = lds_end[base + method.Bkp_ind].reshape(problem.index_map.n.state, problem.index_map.n.nu)

#         Ak[k]   = Ak_bar
#         Bk[k]   = Ak_bar @ Bk_bar
#         Bkp[k]  = Ak_bar @ Bkp_bar

#     z_minus    = np.array(traj_minus_data["z_minus"])

#     return Ak, Bk, Bkp, z_minus

# # Integrate linear system
# def RHS_ltv(tau, lds, nu_ref, t_ref, problem, method):



#     # Initialize
#     lds_dot         = np.zeros_like(lds)
#     N               = method.index_map.N.time_grid

#     # Extract times and FOH control input
#     Om_k            = 1 - tau
#     Om_kp           = tau

#     nrows, ncols    = nu_ref.shape
#     rows            = nrows if nrows > ncols else ncols

#     v_1             = Om_k * np.ones((rows, 1))
#     v_2             = Om_kp * np.ones((rows - 1, 1))
#     Om              = np.diagflat(v_1) + np.diagflat(v_2, 1)

#     nu              = Om @ nu_ref

#     for k in range(N - 1):
#         dt_k       = t_ref[k + 1] - t_ref[k]

#         # Extract state info
#         z_k         = lds[ k * method.lds0_size + method.z_ind ]

#         # Extract continuous time Jacobians
#         fc, Ac, Bc = problem.lin_dyn(tau, z_k, nu[k])

#         # Extract STM
#         Phi_tau     = lds[ k * method.lds0_size + method.Ak_ind ].reshape(problem.index_map.n.state, problem.index_map.n.state)

#         # Construct Jacobians w.r.t. tau
#         f_tau       = dt_k * fc
#         A_tau       = dt_k * Ac
#         B_tau       = dt_k * Om_k * Bc
#         Bp_tau      = dt_k * Om_kp * Bc

#         Phi_tau_inv = np.linalg.pinv(Phi_tau)

#         # Construct derivatives
#         z_dot       = f_tau
#         A_tau_dot   = A_tau @ Phi_tau
#         B_tau_dot   = Phi_tau_inv @ B_tau
#         Bp_tau_dot  = Phi_tau_inv @ Bp_tau

#         # Setup linear system properly
#         lds_dot[ k * method.lds0_size + method.z_ind  ] = z_dot
#         lds_dot[ k * method.lds0_size + method.Ak_ind ] = A_tau_dot.flatten()
#         lds_dot[ k * method.lds0_size + method.Bk_ind ] = B_tau_dot.flatten()
#         lds_dot[ k * method.lds0_size + method.Bkp_ind] = Bp_tau_dot.flatten()

#     return lds_dot


