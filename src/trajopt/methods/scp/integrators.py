from collections.abc import Callable
from typing import TYPE_CHECKING

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from trajopt.core.problem import Problem
from trajopt.utils import tools

if TYPE_CHECKING:
    from trajopt.methods.scp.scvx import SCvx


def _foh(t: jax.Array, t_ref: jax.Array, vals: jax.Array) -> jax.Array:
    """Interpolate vals at t using first-order-hold over the t_ref grid."""
    k = jnp.clip(jnp.searchsorted(t_ref, t, side="right") - 1, 0, t_ref.shape[0] - 2)
    a = (t_ref[k + 1] - t) / (t_ref[k + 1] - t_ref[k])
    return a * vals[k] + (1 - a) * vals[k + 1]


def _barycentric_weights(t_ref: jax.Array) -> jax.Array:
    """Compute barycentric interpolation weights for the nodes t_ref."""
    n    = t_ref.shape[0]
    diff = t_ref[:, None] - t_ref[None, :]
    diff = diff.at[jnp.arange(n), jnp.arange(n)].set(1.0)
    return 1.0 / jnp.prod(diff, axis=1)


def _lagrange(t: jax.Array, t_ref: jax.Array, vals: jax.Array, w: jax.Array) -> jax.Array:
    """Evaluate the barycentric Lagrange interpolant of vals at t."""
    diffs = jnp.where(t - t_ref == 0.0, 1e-30, t - t_ref)
    terms = w / diffs
    return (terms @ vals) / jnp.sum(terms)


def _jit_cached(method: "SCvx", name: str, fn: Callable) -> Callable:
    """Return a JIT-compiled fn cached on method under the given attribute name."""
    if not hasattr(method, name):
        setattr(method, name, jax.jit(fn))
    return getattr(method, name)


def propagate_znu(
    z0: np.ndarray,
    nu: np.ndarray,
    problem: Problem,
    method: "SCvx",
    compiled_attr_name: str = "_jit_propagate_znu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adaptive-step propagation of the augmented state (z, nu) system.

    Returns:
        Tuple of (t_nl, z_nl, nu_nl) sampled on a dense tau grid.

    """
    dynamics    = problem.constraints.get(type="dynamics")[0].fcn
    params      = problem.params
    idx         = problem.index_map.indices
    N           = nu.shape[0]
    use_ps      = getattr(method.flags, "discretize", "ms") == "ps"
    z0_jax      = jnp.asarray(z0)
    nu_jax      = jnp.asarray(nu)
    tau_dense   = jnp.linspace(0.0, 1.0, 1000)
    phase_sched = {k: jnp.asarray(v) for k, v in params.pop("_phase_schedule", {}).items()}

    if use_ps:
        from trajopt.methods.scp.discretize import compute_ps_differentiation_matrix
        _, etau, _, _      = compute_ps_differentiation_matrix(N - 1)
        tau_ref            = jnp.asarray((etau + 1.0) / 2.0)
        bary_w             = _barycentric_weights(tau_ref)
        compiled_attr_name = "_jit_propagate_znu_ps"
    else:
        tau_ref = jnp.linspace(0.0, 1.0, N)
        bary_w  = None

    def interp(tau, nu, tau_ref, bary_w):
        return _lagrange(tau, tau_ref, nu, bary_w) if bary_w is not None else _foh(tau, tau_ref, nu)

    def _solve(z0, nu, tau_ref, tau_dense, bary_w):
        def f_dot(tau, z, _):
            k = jnp.clip(jnp.floor(tau * (N - 1)).astype(jnp.int32), 0, N - 1)
            p = tools.AttrDict({**params, **{key: arr[k] for key, arr in phase_sched.items()}})
            return dynamics(z, interp(tau, nu, tau_ref, bary_w), p)
        return diffrax.diffeqsolve(
            diffrax.ODETerm(f_dot), diffrax.Dopri5(),
            t0=0.0, t1=1.0, dt0=None, y0=z0,
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6),
            saveat=diffrax.SaveAt(ts=tau_dense),
            max_steps=65536,
        )

    sol   = _jit_cached(method, compiled_attr_name, _solve)(z0_jax, nu_jax, tau_ref, tau_dense, bary_w)
    z_nl  = np.asarray(sol.ys)
    nu_nl = np.asarray(jax.vmap(lambda tau: interp(tau, nu_jax, tau_ref, bary_w))(tau_dense))
    t_nl  = z_nl[:, idx.z.time].squeeze(-1)

    return t_nl, z_nl, nu_nl


def propagate_znu_rk4(
    z0: np.ndarray,
    nu: np.ndarray,
    problem: Problem,
    method: "SCvx",
    n_steps: int = 10000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed-step RK4 propagation of the augmented state (z,nu) system."""
    dynamics    = problem.constraints.get(type="dynamics")[0].fcn
    params      = problem.params
    idx         = problem.index_map.indices
    N           = nu.shape[0]
    use_ps      = getattr(method.flags, "discretize", "ms") == "ps"
    z0_jax      = jnp.asarray(z0)
    nu_jax      = jnp.asarray(nu)
    phase_sched = {k: jnp.asarray(v) for k, v in params.pop("_phase_schedule", {}).items()}

    if use_ps:
        from trajopt.methods.scp.discretize import compute_ps_differentiation_matrix
        _, etau, _, _      = compute_ps_differentiation_matrix(N - 1)
        tau_ref            = jnp.asarray((etau + 1.0) / 2.0)
        bary_w             = _barycentric_weights(tau_ref)
    else:
        tau_ref = jnp.linspace(0.0, 1.0, N)
        bary_w  = None

    dt = 1.0 / n_steps
    taus = jnp.linspace(0.0, 1.0, n_steps + 1)

    def interp(tau, nu, tau_ref, bary_w):
        return _lagrange(tau, tau_ref, nu, bary_w) if bary_w is not None else _foh(tau, tau_ref, nu)

    def _solve(z0, nu, tau_ref, bary_w):
        def f(tau, z):
            k = jnp.clip(jnp.floor(tau * (N - 1)).astype(jnp.int32), 0, N - 1)
            p = tools.AttrDict({**params, **{key: arr[k] for key, arr in phase_sched.items()}})
            return dynamics(z, interp(tau, nu, tau_ref, bary_w), p)

        def rk4_step(z, tau):
            k1 = f(tau,          z)
            k2 = f(tau + dt / 2, z + (dt / 2) * k1)
            k3 = f(tau + dt / 2, z + (dt / 2) * k2)
            k4 = f(tau + dt,     z + dt * k3)
            z_next = z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            return z_next, z_next

        _, z_all = jax.lax.scan(rk4_step, z0, taus[:-1])
        return z_all

    solve_fn = _jit_cached(method, "_jit_propagate_znu_rk4", _solve)
    z_all    = np.asarray(solve_fn(z0_jax, nu_jax, tau_ref, bary_w))
    nu_all   = np.asarray(jax.vmap(lambda tau: interp(tau, nu_jax, tau_ref, bary_w))(taus[1:]))
    t_nl     = z_all[:, idx.z.time].squeeze(-1)

    return t_nl, z_all, nu_all


def propagate_txu(
    x0: np.ndarray,
    u: np.ndarray,
    t_ref: np.ndarray,
    t_nl: np.ndarray,
    problem: Problem,
    method: "SCvx",
    compiled_attr_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adaptive-step propagation of physical dynamics dx/dt = f(t, x, u).

    Returns:
        Tuple of (t_nl, x_nl, u_nl) sampled at t_nl.

    """
    dynamics  = problem.constraints.get(type="dynamics")[0].fcn_base
    params    = problem.params
    x0_jax    = jnp.asarray(x0)
    u_jax     = jnp.asarray(u)
    t_ref_jax = jnp.asarray(t_ref)
    t_nl_jax  = jnp.asarray(t_nl)

    def _solve(x0, u, t_ref, t_nl):
        def f_dot(t, x, _):
            return dynamics(t, x, _foh(t, t_ref, u), params)
        return diffrax.diffeqsolve(
            diffrax.ODETerm(f_dot), diffrax.Dopri5(),
            t0=t_ref[0], t1=t_ref[-1], dt0=None, y0=x0,
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6),
            saveat=diffrax.SaveAt(ts=t_nl),
        )

    sol   = _jit_cached(method, compiled_attr_name, _solve)(x0_jax, u_jax, t_ref_jax, t_nl_jax) \
            if compiled_attr_name else _solve(x0_jax, u_jax, t_ref_jax, t_nl_jax)
    u_out = np.asarray(jax.vmap(lambda t: _foh(t, t_ref_jax, u_jax))(t_nl_jax))

    return np.asarray(t_nl_jax), np.asarray(sol.ys), u_out


def propagate_txu_rk4(
    x0: np.ndarray,
    u: np.ndarray,
    t_ref: np.ndarray,
    problem: Problem,
    method: "SCvx",
    n_substeps: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed-step RK4 propagation of physical dynamics dx/dt = f(t, x, u)."""
    dynamics = problem.constraints.get(type="dynamics")[0].fcn_base
    params   = problem.params
    N        = t_ref.shape[0]
    x0_jax   = jnp.asarray(x0)
    u_jax    = jnp.asarray(u)
    t_jax    = jnp.asarray(t_ref)

    def _solve(x0, u, t_ref):
        def propagate_interval(x, k):
            t_k  = t_ref[k]
            t_kp = t_ref[k + 1]
            dt   = (t_kp - t_k) / n_substeps
            ts   = t_k + jnp.arange(n_substeps) * dt

            def rk4_step(x, t):
                u_t = _foh(t, t_ref, u)
                k1 = dynamics(t,          x,                  u_t, params)
                k2 = dynamics(t + dt / 2, x + (dt / 2) * k1, _foh(t + dt/2, t_ref, u), params)
                k3 = dynamics(t + dt / 2, x + (dt / 2) * k2, _foh(t + dt/2, t_ref, u), params)
                k4 = dynamics(t + dt,     x + dt * k3,       _foh(t + dt, t_ref, u), params)
                return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4), None

            x_next, _ = jax.lax.scan(rk4_step, x, ts)
            return x_next, x_next

        _, x_all = jax.lax.scan(propagate_interval, x0, jnp.arange(N - 1))
        return jnp.concatenate([x0[None, :], x_all])

    solve_fn = _jit_cached(method, "_jit_propagate_txu_rk4", _solve)
    x_out    = np.asarray(solve_fn(x0_jax, u_jax, t_jax))
    return np.asarray(t_ref), x_out, np.asarray(u)


def propagate_scipy_rk45(
    x0: np.ndarray,
    u_ref: np.ndarray,
    t_ref: np.ndarray,
    t_nl: np.ndarray,
    problem: Problem,
    method: "SCvx",
    dynamics: Callable | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate physical dynamics dx/dt = f(t, x, u) with SciPy's RK45 solver.

    Returns:
        Tuple of (t, x, u) sampled at t_nl.

    """
    dynamics = problem.constraints.get(type="dynamics")[0].fcn_base if dynamics is None else dynamics
    params   = problem.params
    u_interp = interp1d(t_ref, u_ref, axis=0, fill_value="extrapolate")
    def f_dot(t: float, x: np.ndarray) -> np.ndarray: return np.array(dynamics(t, x, u_interp(t), params))
    sol      = solve_ivp(f_dot, [t_nl[0], t_nl[-1]], x0, t_eval=t_nl, method="RK45", rtol=1e-8, atol=1e-8)
    return sol.t, sol.y.T, u_interp(sol.t)
