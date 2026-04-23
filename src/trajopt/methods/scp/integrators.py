import numpy as np
import jax
import jax.numpy as jnp
import diffrax
import trajopt.utils.tools as tools
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


def _foh(t, t_ref, vals):
    k = jnp.clip(jnp.searchsorted(t_ref, t, side='right') - 1, 0, t_ref.shape[0] - 2)
    a = (t_ref[k + 1] - t) / (t_ref[k + 1] - t_ref[k])
    return a * vals[k] + (1 - a) * vals[k + 1]


def _barycentric_weights(t_ref):
    n    = t_ref.shape[0]
    diff = t_ref[:, None] - t_ref[None, :]
    diff = diff.at[jnp.arange(n), jnp.arange(n)].set(1.0)
    return 1.0 / jnp.prod(diff, axis=1)


def _lagrange(t, t_ref, vals, w):
    diffs = jnp.where(t - t_ref == 0.0, 1e-30, t - t_ref)
    terms = w / diffs
    return (terms @ vals) / jnp.sum(terms)


def _jit_cached(method, name, fn):
    if not hasattr(method, name):
        setattr(method, name, jax.jit(fn))
    return getattr(method, name)


def propagate_znu(z0, nu, problem, method, compiled_attr_name="_jit_propagate_znu"):
    dynamics    = problem.constraints.get(type="dynamics")[0].fcn
    params      = tools.recursive_to_dict(problem.params)
    idx         = problem.index_map.indices
    N           = nu.shape[0]
    use_ps      = getattr(method.flags, 'discretize', 'ms') == 'ps'
    z0_jax      = jnp.asarray(z0)
    nu_jax      = jnp.asarray(nu)
    tau_dense   = jnp.linspace(0.0, 1.0, 1000)
    phase_sched = {k: jnp.asarray(v) for k, v in params.pop('_phase_schedule', {}).items()}

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
            p = {**params, **{key: arr[k] for key, arr in phase_sched.items()}}
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


def propagate_txu(x0, u, t_ref, t_nl, problem, method, compiled_attr_name=None):
    dynamics  = problem.constraints.get(type="dynamics")[0].fcn_base
    params    = tools.recursive_to_dict(problem.params)
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


def propagate_scipy_rk45(x0, u_ref, t_ref, t_nl, problem, method, dynamics=None):
    dynamics = problem.constraints.get(type="dynamics")[0].fcn_base if dynamics is None else dynamics
    params   = tools.recursive_to_dict(problem.params)
    u_interp = interp1d(t_ref, u_ref, axis=0, fill_value="extrapolate")
    def f_dot(t, x): return np.array(dynamics(t, x, u_interp(t), params))
    sol      = solve_ivp(f_dot, [t_nl[0], t_nl[-1]], x0, t_eval=t_nl, method='RK45', rtol=1e-8, atol=1e-8)
    return sol.t, sol.y.T, u_interp(sol.t)
