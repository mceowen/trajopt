import jax
import jax.numpy as jnp
import numpy as np

from trajopt.core.problem import Problem
from trajopt.utils import tools

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


def propagate_rk4(z0, tau0, tau_f, nu_fn, dynamics, params, n_steps=10000):
    solve = _make_rk4_solver(nu_fn, dynamics, params, n_steps)
    taus, z_traj, nu_traj = solve(
        jnp.asarray(z0),
        jnp.asarray(float(tau0)),
        jnp.asarray(float(tau_f)),
    )
    return np.asarray(taus), np.asarray(z_traj), np.asarray(nu_traj)


def propagate_from_nodes(z_nodes, tau_nodes, nu_fn, dynamics, params, n_dense_per_seg=50):
    N   = z_nodes.shape[0]
    n_z = z_nodes.shape[1]

    solve = _make_rk4_solver(nu_fn, dynamics, params, n_dense_per_seg)

    segments = []
    for k in range(N - 1):
        taus_k, z_k, nu_k = solve(
            jnp.asarray(z_nodes[k]),
            jnp.asarray(float(tau_nodes[k])),
            jnp.asarray(float(tau_nodes[k + 1])),
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