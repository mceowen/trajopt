import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Control interpolation
# ---------------------------------------------------------------------------

def interpolate_control_foh(tau_nodes, nu_nodes):
    """First-order hold (linear) control interpolation for multiple shooting.

    Returns a JAX-compatible function nu(z, tau) -> nu_interp.
    """
    tau_ref = jnp.asarray(tau_nodes)
    nu_ref  = jnp.asarray(nu_nodes)
    N_nodes = tau_ref.shape[0]

    def nu_interp(z, tau):
        k = jnp.clip(jnp.searchsorted(tau_ref, tau, side='right') - 1, 0, N_nodes - 2)
        a = (tau - tau_ref[k]) / (tau_ref[k + 1] - tau_ref[k])
        return (1 - a) * nu_ref[k] + a * nu_ref[k + 1]

    return nu_interp


def interpolate_control_lagrange(tau_nodes, nu_nodes):
    """Lagrange polynomial control interpolation for pseudospectral methods.

    Builds the barycentric weights once and evaluates the interpolant at any tau.
    Returns a JAX-compatible function nu(z, tau) -> nu_interp.
    """
    tau_ref = jnp.asarray(tau_nodes)
    nu_ref  = jnp.asarray(nu_nodes)
    N_nodes = tau_ref.shape[0]

    diffs = tau_ref[:, None] - tau_ref[None, :]
    diffs = diffs + jnp.eye(N_nodes)
    bary_weights = 1.0 / jnp.prod(diffs, axis=1)

    def nu_interp(z, tau):
        d = tau - tau_ref
        exact = jnp.argmin(jnp.abs(d))
        is_exact = jnp.abs(d[exact]) < 1e-14

        w_over_d = bary_weights / (d + 1e-300)
        interp_val = jnp.sum(w_over_d[:, None] * nu_ref, axis=0) / jnp.sum(w_over_d)

        return jnp.where(is_exact, nu_ref[exact], interp_val)

    return nu_interp


def make_control_interpolator(discretize, tau_nodes, nu_nodes):
    """Factory that returns the appropriate control interpolator for the method."""
    if discretize == "ps":
        return interpolate_control_lagrange(tau_nodes, nu_nodes)
    else:
        return interpolate_control_foh(tau_nodes, nu_nodes)


# ---------------------------------------------------------------------------
# Full-trajectory propagation (default for analysis)
# ---------------------------------------------------------------------------

def make_trajectory_solver(dynamics, params, n_steps, discretize="ms", hp_segments=1):
    """Create a JIT-compiled RK4 solver that propagates the full trajectory.

    Args:
        dynamics: dynamics function f(z, nu, params)
        params: model parameters
        n_steps: total RK4 integration steps
        discretize: "ms" for FOH interpolation, "ps" for Lagrange
        hp_segments: number of h-intervals (only for ps)
    """
    H = hp_segments
    use_ps = (discretize == "ps")

    if use_ps:
        @jax.jit
        def solve(z0, tau0, tau_f, tau_ref, nu_ref):
            dt   = (tau_f - tau0) / n_steps
            taus = jnp.linspace(tau0, tau_f, n_steps + 1)
            N_nodes = tau_ref.shape[0]
            p = (N_nodes - 1) // H

            boundaries = jnp.zeros(H + 1)
            bary_weights = jnp.zeros((H, p + 1))
            for h in range(H):
                nodes_h = jax.lax.dynamic_slice(tau_ref, (h * p,), (p + 1,))
                boundaries = boundaries.at[h].set(nodes_h[0])
                diffs_h = nodes_h[:, None] - nodes_h[None, :]
                diffs_h = diffs_h + jnp.eye(p + 1)
                bary_weights = bary_weights.at[h].set(1.0 / jnp.prod(diffs_h, axis=1))
            boundaries = boundaries.at[H].set(tau_ref[-1] + 1e-14)

            def nu_interp(z, tau):
                h = jnp.clip(jnp.searchsorted(boundaries, tau, side='right') - 1, 0, H - 1).astype(jnp.int32)
                start = (h * p).astype(jnp.int32)
                zero = jnp.int32(0)
                nodes_h = jax.lax.dynamic_slice(tau_ref, (start,), (p + 1,))
                nu_h = jax.lax.dynamic_slice(nu_ref, (start, zero), (p + 1, nu_ref.shape[1]))
                w_h = bary_weights[h]

                d = tau - nodes_h
                exact = jnp.argmin(jnp.abs(d))
                is_exact = jnp.abs(d[exact]) < 1e-14
                w_over_d = w_h / (d + 1e-300)
                interp_val = jnp.sum(w_over_d[:, None] * nu_h, axis=0) / jnp.sum(w_over_d)
                return jnp.where(is_exact, nu_h[exact], interp_val)

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
                         discretize="ms", n_steps=500, hp_segments=1, _solver=None):
    """Propagate full trajectory from initial condition using interpolated controls.

    For PS: uses piecewise Lagrange interpolation (per h-interval for hp).
    For MS: uses first-order hold (linear interpolation).
    """
    if _solver is None:
        _solver = make_trajectory_solver(dynamics, params, n_steps,
                                         discretize=discretize, hp_segments=hp_segments)

    taus, z_traj, nu_traj = _solver(
        jnp.asarray(z_nodes[0]),
        jnp.asarray(float(tau_nodes[0])),
        jnp.asarray(float(tau_nodes[-1])),
        jnp.asarray(tau_nodes),
        jnp.asarray(nu_nodes),
    )
    return np.asarray(taus), np.asarray(z_traj), np.asarray(nu_traj)


# ---------------------------------------------------------------------------
# Node-by-node propagation (shows defects, useful for MS convergence viz)
# ---------------------------------------------------------------------------

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


def make_node_propagation_solver(dynamics, params, n_steps):
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


def propagate_rk4(z0, tau0, tau_f, nu_fn, dynamics, params, n_steps=1000):
    solve = _make_rk4_solver(nu_fn, dynamics, params, n_steps)
    taus, z_traj, nu_traj = solve(
        jnp.asarray(z0),
        jnp.asarray(float(tau0)),
        jnp.asarray(float(tau_f)),
    )
    return np.asarray(taus), np.asarray(z_traj), np.asarray(nu_traj)


def propagate_from_nodes(z_nodes, tau_nodes, nu_nodes, dynamics, params,
                         n_dense_per_seg=50, _solver=None):
    """Propagate node-by-node (restarts from each node, shows defects)."""
    N   = z_nodes.shape[0]
    n_z = z_nodes.shape[1]

    if _solver is None:
        _solver = make_node_propagation_solver(dynamics, params, n_dense_per_seg)

    tau_ref = jnp.asarray(tau_nodes)
    nu_ref  = jnp.asarray(nu_nodes)

    segments = []
    for k in range(N - 1):
        taus_k, z_k, nu_k = _solver(
            jnp.asarray(z_nodes[k]),
            jnp.asarray(float(tau_nodes[k])),
            jnp.asarray(float(tau_nodes[k + 1])),
            tau_ref,
            nu_ref,
        )
        segments.append((np.asarray(taus_k), np.asarray(z_k), np.asarray(nu_k)))

    n_nu    = segments[0][2].shape[1]
    nan_tau = np.array([np.nan])
    nan_z   = np.full((1, n_z),  np.nan)
    nan_nu  = np.full((1, n_nu), np.nan)

    flat_tau, flat_z, flat_nu = [], [], []
    for k, (tau_k, z_k, nu_k) in enumerate(segments):
        flat_tau.append(tau_k)
        flat_z.append(z_k)
        flat_nu.append(nu_k)
        if k < N - 2:
            flat_tau.append(nan_tau)
            flat_z.append(nan_z)
            flat_nu.append(nan_nu)

    return np.concatenate(flat_tau), np.concatenate(flat_z), np.concatenate(flat_nu)