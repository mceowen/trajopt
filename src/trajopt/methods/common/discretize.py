import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp



# =============================================================================
# SHARED
# =============================================================================

def make_trajectory_solver(dynamics, params, n_steps, discretize="ms",
                           segments=1, control_interpolation="lagrange", dilation_idx=None):
    """JIT-compiled RK4 solver for the full trajectory; "ms" uses FOH controls,
    "ps" uses interpolate_control_ps (lagrange or FOH) on each hp segment."""
    use_ps = (discretize == "ps")

    if use_ps:
        # number of time steps for each hp segment
        n_seg_steps = max(n_steps // segments, 1)

        @jax.jit
        def solve(z0, tau0, tau_f, tau_ref, nu_ref):
            nu_interp = interpolate_control_ps(tau_ref, nu_ref, segments, dilation_idx, control_interpolation)
            knots = jnp.linspace(tau0, tau_f, segments + 1)

            def segment(z_h, h):
                dt   = (knots[h + 1] - knots[h]) / n_seg_steps
                taus = knots[h] + dt * jnp.arange(n_seg_steps)

                # h is passed explicitly, so the step ending on a knot still uses this segment's dilation
                def step(z, tau):
                    nu = nu_interp(tau, h)
                    k1 = dynamics(z,               nu,                        params)
                    k2 = dynamics(z + (dt/2) * k1, nu_interp(tau + dt / 2, h), params)
                    k3 = dynamics(z + (dt/2) * k2, nu_interp(tau + dt / 2, h), params)
                    k4 = dynamics(z + dt * k3,     nu_interp(tau + dt, h),     params)
                    z_next = z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
                    return z_next, (z, nu)

                z_next_h, (z_traj, nu_traj) = jax.lax.scan(step, z_h, taus)
                return z_next_h, (taus, z_traj, nu_traj)

            z_f, (taus, z_traj, nu_traj) = jax.lax.scan(segment, z0, jnp.arange(segments))
            taus    = jnp.concatenate([taus.reshape(-1), tau_f[None]])
            z_traj  = jnp.concatenate([z_traj.reshape(-1, z0.shape[0]), z_f[None]])
            nu_traj = jnp.concatenate([nu_traj.reshape(-1, nu_ref.shape[1]), nu_interp(tau_f, segments - 1)[None]])
            # Stored knot samples belong to the closing segment. Integration above
            # still evaluates every RK stage with its own explicit segment index.
            if dilation_idx is not None:
                p = (tau_ref.shape[0] - 1) // segments
                for h in range(1, segments):
                    nu_traj = nu_traj.at[h * n_seg_steps, dilation_idx].set(
                        nu_ref[(h - 1) * p + 1, dilation_idx])
            return taus, z_traj, nu_traj

    else:
        @jax.jit
        def solve(z0, tau0, tau_f, tau_ref, nu_ref):
            dt   = (tau_f - tau0) / n_steps
            taus = jnp.linspace(tau0, tau_f, n_steps + 1)
            N_nodes = tau_ref.shape[0]

            def nu_interp(z, tau):
                k = jnp.clip(jnp.searchsorted(tau_ref, tau, side='right') - 1, 0, N_nodes - 2)
                a = (tau - tau_ref[k]) / (tau_ref[k + 1] - tau_ref[k])
                return (1 - a) * nu_ref[k] + a * nu_ref[k + 1]

            def step(z, tau):
                nu = nu_interp(z, tau)
                k1 = dynamics(z,               nu,                                        params)
                k2 = dynamics(z + (dt/2) * k1, nu_interp(z + (dt/2) * k1, tau + dt / 2), params)
                k3 = dynamics(z + (dt/2) * k2, nu_interp(z + (dt/2) * k2, tau + dt / 2), params)
                k4 = dynamics(z + dt * k3,     nu_interp(z + dt * k3,     tau + dt),      params)
                z_next = z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
                return z_next, (z, nu)

            z_f, (z_traj, nu_traj) = jax.lax.scan(step, z0, taus[:-1])
            z_traj  = jnp.concatenate([z_traj,  z_f[None]])
            nu_traj = jnp.concatenate([nu_traj, nu_interp(z_f, taus[-1])[None]])
            return taus, z_traj, nu_traj

    return solve


def propagate_trajectory(z_nodes, tau_nodes, nu_nodes, dynamics, params,
                         discretize="ms", n_steps=500, _solver=None,
                         segments=1, control_interpolation="lagrange", dilation_idx=None):
    """Propagate the full trajectory from its initial condition using interpolated controls."""
    if _solver is None:
        _solver = make_trajectory_solver(dynamics, params, n_steps, discretize=discretize, segments=segments,
                                         control_interpolation=control_interpolation, dilation_idx=dilation_idx)

    taus, z_traj, nu_traj = _solver(
        jnp.asarray(z_nodes[0]),
        jnp.asarray(float(tau_nodes[0])),
        jnp.asarray(float(tau_nodes[-1])),
        jnp.asarray(tau_nodes),
        jnp.asarray(nu_nodes),
    )
    return np.asarray(taus), np.asarray(z_traj), np.asarray(nu_traj)


def _make_rk4_solver(nu_fn, dynamics, params, n_steps):

    @jax.jit
    def solve(z0, tau0, tau_f):
        dt   = (tau_f - tau0) / n_steps
        taus = jnp.linspace(tau0, tau_f, n_steps + 1)

        def step(z, tau):
            nu = nu_fn(z, tau)
            k1 = dynamics(z,                nu,                                     params)
            k2 = dynamics(z + (dt/2) * k1,  nu_fn(z + (dt/2) * k1, tau + dt / 2),   params)
            k3 = dynamics(z + (dt/2) * k2,  nu_fn(z + (dt/2) * k2, tau + dt / 2),   params)
            k4 = dynamics(z + dt * k3,      nu_fn(z + dt * k3,     tau + dt),       params)
            z_next = z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            return z_next, (z, nu)

        z_f, (z_traj, nu_traj) = jax.lax.scan(step, z0, taus[:-1])

        z_traj  = jnp.concatenate([z_traj,  z_f[None]])
        nu_traj = jnp.concatenate([nu_traj, nu_fn(z_f, taus[-1])[None]])
        return taus, z_traj, nu_traj

    return solve


def make_node_propagation_solver(dynamics, params, n_steps, discretize="ms",
                                 segments=1, control_interpolation="lagrange", dilation_idx=None):
    use_ps = (discretize == "ps")

    @jax.jit
    def solve(z0, tau0, tau_f, tau_ref, nu_ref, h=0):
        dt   = (tau_f - tau0) / n_steps
        taus = jnp.linspace(tau0, tau_f, n_steps + 1)
        N_nodes = tau_ref.shape[0]

        if use_ps:
            nu_interp_ps = interpolate_control_ps(tau_ref, nu_ref, segments, dilation_idx, control_interpolation)
            nu_interp = lambda z, tau: nu_interp_ps(tau, h)
        else:
            def nu_interp(z, tau):
                k = jnp.clip(jnp.searchsorted(tau_ref, tau, side='right') - 1, 0, N_nodes - 2)
                a = (tau - tau_ref[k]) / (tau_ref[k + 1] - tau_ref[k])
                return (1 - a) * nu_ref[k] + a * nu_ref[k + 1]

        def step(z, tau):
            nu = nu_interp(z, tau)
            k1 = dynamics(z,               nu,                                        params)
            k2 = dynamics(z + (dt/2) * k1, nu_interp(z + (dt/2) * k1, tau + dt / 2), params)
            k3 = dynamics(z + (dt/2) * k2, nu_interp(z + (dt/2) * k2, tau + dt / 2), params)
            k4 = dynamics(z + dt * k3,     nu_interp(z + dt * k3,     tau + dt),      params)
            z_next = z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            return z_next, (z, nu)

        z_f, (z_traj, nu_traj) = jax.lax.scan(step, z0, taus[:-1])
        z_traj  = jnp.concatenate([z_traj,  z_f[None]])
        nu_traj = jnp.concatenate([nu_traj, nu_interp(z_f, taus[-1])[None]])
        if use_ps and dilation_idx is not None:
            p = (N_nodes - 1) // segments
            closing_h = jnp.maximum(h - 1, 0)
            nu_traj = nu_traj.at[0, dilation_idx].set(nu_ref[closing_h * p + 1, dilation_idx])
        return taus, z_traj, nu_traj

    return solve


def propagate_rk4(z0, tau0, tau_f, nu_fn, dynamics, params, n_steps=1000):
    solve = _make_rk4_solver(nu_fn, dynamics, params, n_steps)
    taus, z_traj, nu_traj = solve(
        jnp.asarray(z0),
        jnp.asarray(float(tau0)),
        jnp.asarray(float(tau_f)),
    )
    return np.asarray(taus), np.asarray(z_traj), np.asarray(nu_traj)


def propagate_from_nodes(z_nodes, tau_nodes, nu_nodes, dynamics, params,
                         n_dense_per_seg=50, _solver=None, discretize="ms",
                         segments=1, control_interpolation="lagrange", dilation_idx=None):
    """Propagate node-by-node (restarts from each node, shows defects); "ps" restarts from each hp segment knot."""
    N   = z_nodes.shape[0]
    n_z = z_nodes.shape[1]

    if _solver is None:
        _solver = make_node_propagation_solver(dynamics, params, n_dense_per_seg, discretize=discretize, segments=segments,
                                               control_interpolation=control_interpolation, dilation_idx=dilation_idx)

    tau_ref = jnp.asarray(tau_nodes)
    nu_ref  = jnp.asarray(nu_nodes)

    # restart stride: every node for ms, every segment for ps (collocation nodes are not shooting nodes)
    stride = (N - 1) // segments if discretize == "ps" else 1

    phases = []
    for k in range(0, N - 1, stride):
        taus_k, z_k, nu_k = _solver(
            jnp.asarray(z_nodes[k]),
            jnp.asarray(float(tau_nodes[k])),
            jnp.asarray(float(tau_nodes[k + stride])),
            tau_ref,
            nu_ref,
            k // stride,
        )
        phases.append((np.asarray(taus_k), np.asarray(z_k), np.asarray(nu_k)))

    n_nu    = phases[0][2].shape[1]
    nan_tau = np.array([np.nan])
    nan_z   = np.full((1, n_z),  np.nan)
    nan_nu  = np.full((1, n_nu), np.nan)

    flat_tau, flat_z, flat_nu = [], [], []
    for k, (tau_k, z_k, nu_k) in enumerate(phases):
        flat_tau.append(tau_k)
        flat_z.append(z_k)
        flat_nu.append(nu_k)
        if k < len(phases) - 1:
            flat_tau.append(nan_tau)
            flat_z.append(nan_z)
            flat_nu.append(nan_nu)

    return np.concatenate(flat_tau), np.concatenate(flat_z), np.concatenate(flat_nu)


# =============================================================================
# MULTIPLE SHOOTING
# =============================================================================

# ---------------------------------------------------------------------------
# time representation (subproblem-level: cp_params.tau/ps_t_offset, dt/ds constraints)
# ---------------------------------------------------------------------------

def create_time_params_ms(subproblem):
    pass


def update_time_params_ms(subproblem):
    pass


def create_time_constraints_ms(subproblem):
    N = subproblem.index_map.N.all
    for k in range(N - 1):
        t_0 = subproblem.t_ref[0, 0] + subproblem.dt[0, 0]
        t_1 = subproblem.t_ref[1, 0] + subproblem.dt[1, 0]

        t_k  = subproblem.t_ref[k, 0] + subproblem.dt[k, 0]
        t_kp = subproblem.t_ref[k + 1, 0] + subproblem.dt[k + 1, 0]

        s_k  = subproblem.s_ref[k, 0] + subproblem.ds[k, 0]
        s_kp = subproblem.s_ref[k + 1, 0] + subproblem.ds[k + 1, 0]

        subproblem.cp_constraints.append(t_k >= 0)
        # a negative dilation puts NaNs in downstream nonlinear terms (e.g. aero)
        subproblem.cp_constraints.append(0.0 <= s_k)

        if hasattr(subproblem.hyperparams.discretize, "equal_dt") and bool(subproblem.hyperparams.discretize.equal_dt):
            interval_k = t_kp - t_k
            interval_0 = t_1 - t_0
            subproblem.cp_constraints.append(interval_k == interval_0)

        if hasattr(subproblem.hyperparams.flags, "zoh_dilation") and bool(subproblem.hyperparams.flags.zoh_dilation):
            subproblem.cp_constraints.append(s_k == s_kp)

    subproblem.cp_constraints.append(0.0 <= subproblem.s_ref[N - 1, 0] + subproblem.ds[N - 1, 0])


def final_time_interval_constraints_ms(constraint, subproblem):
    N = subproblem.index_map.N.all
    for k in range(N - 1):
        t_k          = subproblem.t_ref[k, 0] + subproblem.dt[k, 0]
        t_kp         = subproblem.t_ref[k + 1, 0] + subproblem.dt[k + 1, 0]
        t_interval_k = t_kp - t_k
        subproblem.cp_constraints.append(t_interval_k <= subproblem.cp_params.dt_max)
        subproblem.cp_constraints.append(t_interval_k >= subproblem.cp_params.dt_min)


# ---------------------------------------------------------------------------
# dynamics constraint lifecycle (first-order)
# ---------------------------------------------------------------------------

def dynamics_compile_ms(constraint, subproblem):
    dyn_fcn = constraint.dyn_fcn
    N_grid  = subproblem.index_map.N.all
    nsub    = int(getattr(subproblem.hyperparams.discretize, 'nsub', 10))

    _, propagate_k, _ = make_rk4_propagator(dyn_fcn, N_grid, nsub)
    prop_jacobians_k = jax.jacfwd(propagate_k, argnums=(1, 2, 3))

    constraint.propagate           = jax.jit(jax.vmap(propagate_k,      in_axes=(0, 0, 0, 0, None)))
    constraint.propagate_jacobians = jax.jit(jax.vmap(prop_jacobians_k, in_axes=(0, 0, 0, 0, None)))


def dynamics_create_cvxpy_parameters_ms(constraint, subproblem):
    N, n_z, n_nu = subproblem.index_map.N.all, subproblem.index_map.n.z, subproblem.index_map.n.nu
    subproblem.cp_params.Ak  = cp.Parameter((N - 1, n_z, n_z),  name="Ak")
    subproblem.cp_params.Bk  = cp.Parameter((N - 1, n_z, n_nu), name="Bk")
    subproblem.cp_params.Bkp = cp.Parameter((N - 1, n_z, n_nu), name="Bkp")
    subproblem.cp_params.z_m = cp.Parameter((N, n_z), name="z_minus")


def dynamics_create_cvxpy_constraints_ms(constraint, subproblem):
    N = subproblem.index_map.N.all
    vb_dyn = constraint.penalties.vb_var
    subproblem.cp_dyn_constraints = []

    for k in range(N - 1):
        dz_k   = subproblem.dz[k]
        dnu_k  = subproblem.dnu[k]
        dnu_kp = subproblem.dnu[k + 1]

        Ak  = subproblem.cp_params.Ak[k]
        Bk  = subproblem.cp_params.Bk[k]
        Bkp = subproblem.cp_params.Bkp[k]

        rhs = Ak @ dz_k + Bk @ dnu_k + Bkp @ dnu_kp

        z_ref_prop_kp = subproblem.cp_params.z_m[k + 1]
        lhs = subproblem.dz[k + 1] + subproblem.cp_params.z_ref[k + 1]
        rhs_full = z_ref_prop_kp + rhs + (vb_dyn[k] if vb_dyn is not None else 0)

        cnst = (lhs == rhs_full)
        subproblem.cp_dyn_constraints.append(cnst)
        subproblem.cp_constraints.append(cnst)


def dynamics_update_cvxpy_parameters_ms(constraint, subproblem):
    z_opt  = subproblem.current_iter_data.z_opt
    nu_opt = subproblem.current_iter_data.nu_opt

    z_ref_ks   = jnp.asarray(z_opt[:-1])
    nu_ref_ks  = jnp.asarray(nu_opt[:-1])
    nu_ref_kps = jnp.asarray(nu_opt[1:])
    params     = subproblem.params
    ks = jnp.arange(subproblem.index_map.N.all - 1)

    z_minus              = constraint.propagate(ks, z_ref_ks, nu_ref_ks, nu_ref_kps, params)
    A_jax, B_jax, Bp_jax = constraint.propagate_jacobians(ks, z_ref_ks, nu_ref_ks, nu_ref_kps, params)

    z_ref_0 = z_ref_ks[[0], :]
    subproblem.cp_params.Ak.value  = np.asarray(A_jax)
    subproblem.cp_params.Bk.value  = np.asarray(B_jax)
    subproblem.cp_params.Bkp.value = np.asarray(Bp_jax)
    subproblem.cp_params.z_m.value = np.asarray(jnp.vstack([z_ref_0, z_minus]))


def dynamics_update_current_iter_data_ms(constraint, subproblem):
    z_opt  = subproblem.current_iter_data.z_opt
    nu_opt = subproblem.current_iter_data.nu_opt
    ks         = jnp.arange(subproblem.index_map.N.all - 1)
    z_ref_ks   = jnp.asarray(z_opt[:-1])
    nu_ref_ks  = jnp.asarray(nu_opt[:-1])
    nu_ref_kps = jnp.asarray(nu_opt[1:])
    z_minus    = np.asarray(constraint.propagate(ks, z_ref_ks, nu_ref_ks, nu_ref_kps, subproblem.params))
    subproblem.current_iter_data.defect = z_opt[1:] - z_minus


# ---------------------------------------------------------------------------
# dynamics constraint lifecycle (second-order -- adds Hessian bookkeeping)
# ---------------------------------------------------------------------------

def dynamics_compile_second_order_ms(constraint, subproblem):
    """Extends dynamics_compile_ms with the Lagrangian-Hessian machinery a second-order trust region needs."""
    dynamics_compile_ms(constraint, subproblem)

    N_grid = subproblem.index_map.N.all
    nsub   = int(getattr(subproblem.hyperparams.discretize, 'nsub', 10))

    lagrangian_propagate_k = make_lagrangian_rk4_propagator(constraint.dyn_fcn, N_grid, nsub)
    cnstr_hessians_k = jax.hessian(lagrangian_propagate_k, argnums=(2, 3, 4))
    constraint.cnstr_hessians = jax.jit(jax.vmap(cnstr_hessians_k, in_axes=(0, 0, 0, 0, 0, None)))


def dynamics_create_cvxpy_parameters_second_order_ms(constraint, subproblem):
    dynamics_create_cvxpy_parameters_ms(constraint, subproblem)
    N, n_z, n_nu = subproblem.index_map.N.all, subproblem.index_map.n.z, subproblem.index_map.n.nu
    n_w = n_z + n_nu
    subproblem.cp_params.L = cp.Parameter((N, n_w, n_w), name="L")


def dynamics_update_cvxpy_parameters_second_order_ms(constraint, subproblem):
    dynamics_update_cvxpy_parameters_ms(constraint, subproblem)

    z_opt      = subproblem.current_iter_data.z_opt
    nu_opt     = subproblem.current_iter_data.nu_opt
    z_ref_ks   = jnp.asarray(z_opt[:-1])
    nu_ref_ks  = jnp.asarray(nu_opt[:-1])
    nu_ref_kps = jnp.asarray(nu_opt[1:])
    ks = jnp.arange(subproblem.index_map.N.all - 1)

    lam_refs = jnp.asarray(constraint.lagrangian_dual)
    constraint.H_z_k, constraint.H_nu_k, constraint.H_nu_kp = constraint.cnstr_hessians(
        ks, lam_refs, z_ref_ks, nu_ref_ks, nu_ref_kps, subproblem.params,
    )


# ---------------------------------------------------------------------------
# RK4 propagation over one node interval, FOH-interpolating control in between
# ---------------------------------------------------------------------------

def make_rk4_propagator(dyn_fcn, N_grid, nsub):
    """RK4 sub-step propagator over node interval [k, k+1], control FOH-interpolated between nu_k and nu_kp."""
    delta_tau = 1.0 / (N_grid - 1)
    dt_rk4    = delta_tau / nsub

    def f_dot(k, tau, z, nu_k, nu_kp, params):
        tau_k  = k / (N_grid - 1)
        tau_kp = (k + 1) / (N_grid - 1)
        a      = (tau_kp - tau) / (tau_kp - tau_k)
        b      = (tau - tau_k)  / (tau_kp - tau_k)
        nu     = a * nu_k + b * nu_kp
        return dyn_fcn(z, nu, params)

    def rk4_step(carry, tau):
        z, k, nu_k, nu_kp, params = carry
        k1 = f_dot(k, tau,            z,                   nu_k, nu_kp, params)
        k2 = f_dot(k, tau + dt_rk4/2, z + (dt_rk4/2) * k1, nu_k, nu_kp, params)
        k3 = f_dot(k, tau + dt_rk4/2, z + (dt_rk4/2) * k2, nu_k, nu_kp, params)
        k4 = f_dot(k, tau + dt_rk4,   z +     dt_rk4 * k3, nu_k, nu_kp, params)
        z_next = z + (dt_rk4 / 6) * (k1 + 2*k2 + 2*k3 + k4)
        return (z_next, k, nu_k, nu_kp, params), None

    def propagate_k(k, z_k, nu_k, nu_kp, params):
        tau_k = k / (N_grid - 1)
        taus  = tau_k + jnp.arange(nsub) * dt_rk4
        carry_init = (z_k, k, nu_k, nu_kp, params)
        (z_kp, _, _, _, _), _ = jax.lax.scan(rk4_step, carry_init, taus)
        return z_kp

    return f_dot, propagate_k, dt_rk4


def make_lagrangian_rk4_propagator(dyn_fcn, N_grid, nsub):
    """Like make_rk4_propagator, but also accumulates the running Lagrangian for Hessian autodiff."""
    f_dot, _, dt_rk4 = make_rk4_propagator(dyn_fcn, N_grid, nsub)

    def scalar_f_dot(lam_k, k, tau, z, nu_k, nu_kp, params):
        return -lam_k @ f_dot(k, tau, z, nu_k, nu_kp, params)

    def lagrangian_rk4_step(carry, tau):
        z, L_val, lam_k, k, nu_k, nu_kp, params = carry
        k1_z = f_dot(k, tau,            z,                      nu_k, nu_kp, params)
        k2_z = f_dot(k, tau + dt_rk4/2, z + (dt_rk4/2) * k1_z, nu_k, nu_kp, params)
        k3_z = f_dot(k, tau + dt_rk4/2, z + (dt_rk4/2) * k2_z, nu_k, nu_kp, params)
        k4_z = f_dot(k, tau + dt_rk4,   z +     dt_rk4  * k3_z, nu_k, nu_kp, params)
        z_next = z + (dt_rk4 / 6) * (k1_z + 2*k2_z + 2*k3_z + k4_z)
        k1_L = scalar_f_dot(lam_k, k, tau,            z,                      nu_k, nu_kp, params)
        k2_L = scalar_f_dot(lam_k, k, tau + dt_rk4/2, z + (dt_rk4/2) * k1_z, nu_k, nu_kp, params)
        k3_L = scalar_f_dot(lam_k, k, tau + dt_rk4/2, z + (dt_rk4/2) * k2_z, nu_k, nu_kp, params)
        k4_L = scalar_f_dot(lam_k, k, tau + dt_rk4,   z +     dt_rk4  * k3_z, nu_k, nu_kp, params)
        L_next = L_val + (dt_rk4 / 6) * (k1_L + 2*k2_L + 2*k3_L + k4_L)
        return (z_next, L_next, lam_k, k, nu_k, nu_kp, params), None

    def lagrangian_propagate_k(k, lam_k, z_k, nu_k, nu_kp, params):
        tau_k = k / (N_grid - 1)
        taus  = tau_k + jnp.arange(nsub) * dt_rk4
        carry_init = (z_k, 0.0, lam_k, k, nu_k, nu_kp, params)
        (_, L_final, _, _, _, _, _), _ = jax.lax.scan(lagrangian_rk4_step, carry_init, taus)
        return L_final

    return lagrangian_propagate_k


# =============================================================================
# PSEUDOSPECTRAL
# =============================================================================

# ---------------------------------------------------------------------------
# time representation (subproblem-level: dt/ds constraints; time is affine per segment through the dynamics)
# ---------------------------------------------------------------------------

def create_time_params_ps(subproblem):
    pass


def update_time_params_ps(subproblem):
    pass


def create_time_constraints_ps(subproblem):
    N = subproblem.index_map.N.all
    H = subproblem.mesh.segments
    p = (N - 1) // H

    for k in range(N - 1):
        s_k  = subproblem.s_ref[k, 0] + subproblem.ds[k, 0]
        s_kp = subproblem.s_ref[k + 1, 0] + subproblem.ds[k + 1, 0]

        subproblem.cp_constraints.append(0.0 <= s_k)

        # apply consecutive dilations equality when: p method (H=1), inside the h segments, zoh_dilation = 1
        if k == 0 or k % p or (hasattr(subproblem.hyperparams.flags, "zoh_dilation") and bool(subproblem.hyperparams.flags.zoh_dilation)):
            subproblem.cp_constraints.append(s_k == s_kp)

    if hasattr(subproblem.hyperparams.discretize, "equal_dt") and bool(subproblem.hyperparams.discretize.equal_dt):
        # equal H segments widths
        width_0 = (subproblem.t_ref[p, 0] + subproblem.dt[p, 0]) - (subproblem.t_ref[0, 0] + subproblem.dt[0, 0])
        for h in range(1, H):
            t_h  = subproblem.t_ref[h * p, 0] + subproblem.dt[h * p, 0]
            t_hp = subproblem.t_ref[(h + 1) * p, 0] + subproblem.dt[(h + 1) * p, 0]
            subproblem.cp_constraints.append(t_hp - t_h == width_0)

    subproblem.cp_constraints.append(subproblem.t_ref[N - 1, 0] + subproblem.dt[N - 1, 0] >= 0.0)
    subproblem.cp_constraints.append(0.0 <= subproblem.s_ref[N - 1, 0] + subproblem.ds[N - 1, 0])


def final_time_interval_constraints_ps(constraint, subproblem):
    # segment widths treated like the ms node intervals
    N = subproblem.index_map.N.all
    H = subproblem.mesh.segments
    p = (N - 1) // H
    for h in range(H):
        t_h  = subproblem.t_ref[h * p, 0] + subproblem.dt[h * p, 0]
        t_hp = subproblem.t_ref[(h + 1) * p, 0] + subproblem.dt[(h + 1) * p, 0]
        subproblem.cp_constraints.append(t_hp - t_h <= subproblem.cp_params.T_max / H)
        subproblem.cp_constraints.append(t_hp - t_h >= subproblem.cp_params.T_min / H)


# ---------------------------------------------------------------------------
# dynamics constraint lifecycle (first-order)
# ---------------------------------------------------------------------------

def dynamics_compile_ps(constraint, subproblem):
    mesh = subproblem.mesh
    constraint.ps_D = mesh.differentiation
    # map the ps defects from derivatives to states (now comparable with ms)
    p = (len(mesh.tau) - 1) // mesh.segments
    constraint.ps_Dinv = sp.linalg.block_diag(*[np.linalg.inv(2.0 * mesh.differentiation[:p, 1:p + 1])] * mesh.segments)
    constraint.dyn_fcn_batched = jax.jit(jax.vmap(constraint.dyn_fcn, in_axes=(0, 0, None)))

    # only ps's per-node constraint linearization needs the dynamics Jacobians
    df_dz  = jax.jit(jax.jacfwd(constraint.constraint.fcn_znu, argnums=0))
    df_dnu = jax.jit(jax.jacfwd(constraint.constraint.fcn_znu, argnums=1))
    dyn_fcn = constraint.dyn_fcn
    constraint.lin_dyn = lambda z, nu, params: (dyn_fcn(z, nu, params), df_dz(z, nu, params), df_dnu(z, nu, params))


def dynamics_create_cvxpy_parameters_ps(constraint, subproblem):
    N, n_z, n_nu = subproblem.index_map.N.all, subproblem.index_map.n.z, subproblem.index_map.n.nu
    N_col = N - 1
    subproblem.cp_params.ps_f_ref = cp.Parameter((N_col, n_z),       name="ps_f_ref")
    subproblem.cp_params.ps_Ac    = cp.Parameter((N_col, n_z, n_z),  name="ps_Ac")
    subproblem.cp_params.ps_Bc    = cp.Parameter((N_col, n_z, n_nu), name="ps_Bc")


def dynamics_create_cvxpy_constraints_ps(constraint, subproblem):
    N_col  = subproblem.index_map.N.all - 1
    vb_dyn = constraint.penalties.vb_var
    D = constraint.ps_D

    subproblem.cp_dyn_constraints = []

    Z = subproblem.cp_params.z_ref + subproblem.dz
    lhs = 2.0 * (D @ Z)

    for k in range(N_col):
        rhs_k = (subproblem.cp_params.ps_f_ref[k]
                 + subproblem.cp_params.ps_Ac[k] @ subproblem.dz[k + 1]
                 + subproblem.cp_params.ps_Bc[k] @ subproblem.dnu[k + 1]
                 + (vb_dyn[k] if vb_dyn is not None else 0))
        cnst = (lhs[k] == rhs_k)
        subproblem.cp_dyn_constraints.append(cnst)
        subproblem.cp_constraints.append(cnst)


def dynamics_update_cvxpy_parameters_ps(constraint, subproblem):
    z_opt  = subproblem.current_iter_data.z_opt
    nu_opt = subproblem.current_iter_data.nu_opt
    N_col  = subproblem.index_map.N.all - 1
    n_z    = subproblem.index_map.n.z
    n_nu   = subproblem.index_map.n.nu
    params = subproblem.params

    f_ref_col = np.zeros((N_col, n_z))
    Ac_col    = np.zeros((N_col, n_z, n_z))
    Bc_col    = np.zeros((N_col, n_z, n_nu))

    for k in range(N_col):
        z_k  = np.asarray(z_opt[k + 1])
        nu_k = np.asarray(nu_opt[k + 1])
        fc_k, Ac_k, Bc_k = constraint.lin_dyn(z_k, nu_k, params)
        f_ref_col[k, :] = np.asarray(fc_k)
        Ac_col[k, :, :] = np.asarray(Ac_k)
        Bc_col[k, :, :] = np.asarray(Bc_k)

    subproblem.cp_params.ps_f_ref.value = f_ref_col
    subproblem.cp_params.ps_Ac.value    = Ac_col
    subproblem.cp_params.ps_Bc.value    = Bc_col


def dynamics_update_current_iter_data_ps(constraint, subproblem):
    z_opt  = subproblem.current_iter_data.z_opt
    nu_opt = subproblem.current_iter_data.nu_opt
    z_jnp  = jnp.asarray(z_opt)
    nu_jnp = jnp.asarray(nu_opt)
    D_jnp  = jnp.asarray(constraint.ps_D)

    lhs = 2.0 * D_jnp @ z_jnp

    f_vals = constraint.dyn_fcn_batched(z_jnp[1:], nu_jnp[1:], subproblem.params)
    subproblem.current_iter_data.defect = constraint.ps_Dinv @ np.asarray(lhs - f_vals)


# ---------------------------------------------------------------------------
# dynamics constraint lifecycle (second-order -- adds Hessian bookkeeping)
# ---------------------------------------------------------------------------

def dynamics_compile_second_order_ps(constraint, subproblem):
    """ps counterpart of dynamics_compile_second_order_ms."""
    dynamics_compile_ps(constraint, subproblem)

    dyn_fcn = constraint.dyn_fcn

    def ps_lagrangian_k(lam_k, z_k, nu_k, params):
        return -lam_k @ dyn_fcn(z_k, nu_k, params)

    constraint.ps_cnstr_hessians = jax.jit(jax.vmap(
        jax.hessian(ps_lagrangian_k, argnums=(1, 2)),
        in_axes=(0, 0, 0, None),
    ))


def dynamics_create_cvxpy_parameters_second_order_ps(constraint, subproblem):
    dynamics_create_cvxpy_parameters_ps(constraint, subproblem)
    N, n_z, n_nu = subproblem.index_map.N.all, subproblem.index_map.n.z, subproblem.index_map.n.nu
    n_w = n_z + n_nu
    subproblem.cp_params.L = cp.Parameter((N, n_w, n_w), name="L")


def dynamics_update_cvxpy_parameters_second_order_ps(constraint, subproblem):
    dynamics_update_cvxpy_parameters_ps(constraint, subproblem)

    z_opt  = subproblem.current_iter_data.z_opt
    nu_opt = subproblem.current_iter_data.nu_opt
    lam_refs = jnp.asarray(constraint.lagrangian_dual)
    z_col    = jnp.asarray(z_opt[1:])
    nu_col   = jnp.asarray(nu_opt[1:])
    constraint.ps_H_z, constraint.ps_H_nu = constraint.ps_cnstr_hessians(lam_refs, z_col, nu_col, subproblem.params)


# ---------------------------------------------------------------------------
# control interpolation
# ---------------------------------------------------------------------------

def interpolate_control_ps(tau_nodes, nu_nodes, segments, dilation_idx, kind="lagrange"):
    """Returns a nu(tau, h) interpolator on hp segment h: controls are Lagrange on the segment's
    collocation nodes ("lagrange") or linear between nodes ("foh"), while dilation is constant per segment."""
    if kind not in ('lagrange', 'foh'):
        raise ValueError(f"Unknown pseudospectral control interpolation {kind!r}")
    tau_ref = jnp.asarray(tau_nodes)
    nu_ref  = jnp.asarray(nu_nodes)
    N_nodes = tau_ref.shape[0]
    p       = (N_nodes - 1) // segments

    # equal order segments, so they share the barycentric weights
    tau_loc = tau_ref[1:p + 1]
    diffs   = tau_loc[:, None] - tau_loc[None, :] + jnp.eye(p)
    bary_weights = 1.0 / jnp.prod(diffs, axis=1)

    def nu_interp(tau, h=None):
        if h is None:
            h = jnp.clip(jnp.searchsorted(tau_ref[p::p], tau, side='left'), 0, segments - 1)
        h = jnp.asarray(h, dtype=jnp.int32)
        if kind == "foh":
            k  = jnp.clip(jnp.searchsorted(tau_ref, tau, side='right') - 1, 0, N_nodes - 2)
            a  = (tau - tau_ref[k]) / (tau_ref[k + 1] - tau_ref[k])
            nu = (1 - a) * nu_ref[k] + a * nu_ref[k + 1]
        else:
            # controls only the segment's collocation nodes, start node belongs to the previous segment
            nodes = jax.lax.dynamic_slice(tau_ref, (h * p + 1,), (p,))
            vals  = jax.lax.dynamic_slice(nu_ref, (h * p + 1, jnp.int32(0)), (p, nu_ref.shape[1]))
            d     = tau - nodes
            exact = jnp.argmin(jnp.abs(d))
            w_over_d = bary_weights / (d + 1e-300)
            interp_val = jnp.sum(w_over_d[:, None] * vals, axis=0) / jnp.sum(w_over_d)
            nu = jnp.where(jnp.abs(d[exact]) < 1e-14, vals[exact], interp_val)

        # the transcription holds the dilation constant on each segment's collocation nodes
        if dilation_idx is not None:
            nu = nu.at[dilation_idx].set(nu_ref[h * p + 1, dilation_idx])
        return nu

    return nu_interp


USE_SPARTAN = True
USE_HARDCODED_NEWTON = False

# ---------------------------------------------------------------------
# Legendre polynomial helper
# ---------------------------------------------------------------------
def compute_legendre(N: int, x: np.ndarray, use_spartan: bool = USE_SPARTAN) -> tuple[np.ndarray, np.ndarray]:
    """Legendre polynomial P_n(x) and its derivative, via SPARTAN recursion or scipy."""
    x = np.asarray(x, dtype=float)

    if use_spartan:
        x = np.asarray(x, dtype=float)

        if N == 0:
            return np.ones_like(x), np.zeros_like(x)
        if N == 1:
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

    if N == 0:
        return np.ones_like(x), np.zeros_like(x)

    Pn      = sp.special.eval_legendre(N, x)
    Pnm1    = sp.special.eval_legendre(N - 1, x)

    dPn     = N * (x * Pn - Pnm1) / (x**2 - 1.0)

    return Pn, dPn

# ---------------------------------------------------------------------
# Flipped Legendre-Radau polynomial
# ---------------------------------------------------------------------
# def flipped_radau_polynomial(N: int, tau: np.ndarray, use_spartan: bool = USE_SPARTAN) -> np.ndarray:
#     """Evaluate the flipped Legendre-Radau polynomial R_n(tau) = P_n(tau) - P_{n-1}(tau)."""
#     tau     = np.asarray(tau, dtype=float)

#     Ln, _   = compute_legendre(N, tau, use_spartan=use_spartan)
#     Lnm1, _ = compute_legendre(N - 1, tau, use_spartan=use_spartan)

#     return Ln - Lnm1


# def flipped_radau_polynomial_derivative(N: int, tau: np.ndarray) -> np.ndarray:
#     """Derivative of the flipped Legendre-Radau polynomial."""
#     tau         = np.asarray(tau, dtype=float)
#     _, dLn      = compute_legendre(N, tau, use_spartan=USE_SPARTAN)
#     _, dLnm1    = compute_legendre(N - 1, tau, use_spartan=USE_SPARTAN)

#     return dLn - dLnm1


# ---------------------------------------------------------------------
# Compute flipped Radau nodes and quadrature weights
# ---------------------------------------------------------------------
def flipped_radau_nodes_and_weights(N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (tau, etau, w): collocation nodes, full node set (includes -1), and quadrature weights."""
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
# ---------------------------------------------------------------------

def differentiation_matrix(etau: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """D maps state values at the full node set (incl. -1) to derivatives at the collocation nodes."""
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


def flipped_radau_differential_operator(N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (tau, etau, w, D): fLGR nodes, full node set, quadrature weights, and differential operator."""
    tau, etau, w    = flipped_radau_nodes_and_weights(N)
    D, _, _         = differentiation_matrix(etau)

    return tau, etau, w, D


# ---------------------------------------------------------------------
# hp composite differential operator
# ---------------------------------------------------------------------

def flipped_radau_hp_operator(N_col: int, H: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """hp-composite fLGR on H segments of equal order; output D is the global (N_col, N_col+1) staircase of local blocks."""
    if N_col % H != 0:
        raise ValueError(f"N_col={N_col} must be divisible by H={H}")
    if H == 1:
        return flipped_radau_differential_operator(N_col)

    p = N_col // H
    _, etau_local, w_local, D_local = flipped_radau_differential_operator(p)

    # segment h spans [-1 + 2h/H, -1 + 2(h+1)/H], consecutive segments share their final-to-initial node
    etau = np.concatenate([[-1.0]] + [-1.0 + (2.0 * h + etau_local[1:] + 1.0) / H for h in range(H)])
    w    = np.tile(w_local, H) / H

    D = np.zeros((N_col, N_col + 1))
    for h in range(H):
        D[h * p:(h + 1) * p, h * p:(h + 1) * p + 1] = H * D_local

    return etau[1:], etau, w, D
