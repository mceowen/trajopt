import numpy as np
import jax
import jax.numpy as jnp
import diffrax
import trajopt.utils.tools as tools
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

def _call_dynamics(dynamics, t, y, v, params):
    try:
        return dynamics(t, y, v, params)
    except TypeError:
        return dynamics(y, v, params)


def compile_dense_jax_propagator(problem, method, params, dynamics=None, compiled_attr_name="propagate_rk4_dense_jit"):

    dynamics = problem.constraints.get(type="dynamics")[0].fcn if dynamics is None else dynamics

    def rk4_step(yi, ti, dt, v_ref, t_ref, params):
        k1 = y_dot(yi, ti, v_ref, t_ref, params)
        k2 = y_dot(yi + 0.5*dt*k1, ti + 0.5*dt, v_ref, t_ref, params)
        k3 = y_dot(yi + 0.5*dt*k2, ti + 0.5*dt, v_ref, t_ref, params)
        k4 = y_dot(yi + dt*k3, ti + dt, v_ref, t_ref, params)
        return yi + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def y_dot(y, t, v_ref, t_ref, params):
        k = jnp.searchsorted(t_ref, t, side='right') - 1
        k = jnp.clip(k, 0, len(t_ref) - 2)

        v_ref_k = v_ref[k]
        v_ref_kp = v_ref[k+1]
        t_ref_k = t_ref[k]
        t_ref_kp = t_ref[k+1]

        # FOH interpolation of controls
        a = 1 - (t - t_ref_k) / (t_ref_kp - t_ref_k)
        b = (t - t_ref_k) / (t_ref_kp - t_ref_k)
        v = a * v_ref_k + b * v_ref_kp

        return _call_dynamics(dynamics, t, y, v, params)

    rk4_step_jit = rk4_step

    def propagate(y0, v_ref, t_ref, t_nl, params):
        dt = t_nl[1] - t_nl[0]

        def rk4_step_partial(yi, ti):
            yi_next = rk4_step_jit(yi, ti, dt, v_ref, t_ref, params)
            return yi_next, yi_next

        yf, ys = jax.lax.scan(rk4_step_partial, y0, t_nl[:-1])

        y_full = jnp.vstack([y0[None, :], ys])

        return y_full

    setattr(method, compiled_attr_name, jax.jit(propagate))

def propagate_jax_rk4_dense(z0, nu_ref, t_ref, t_nl, problem, method, compiled_attr_name="propagate_rk4_dense_jit"):

    z0_jax = jnp.array(z0)
    nu_ref_jax = jnp.array(nu_ref)
    t_ref_jax = jnp.array(t_ref)
    t_nl_jax = jnp.array(t_nl)
    
    params = problem.params
    params_jax = tools.recursive_to_dict(params)

    propagate_jit = getattr(method, compiled_attr_name)
    z_full_jax = propagate_jit(z0_jax, nu_ref_jax, t_ref_jax, t_nl_jax, params_jax)

    t_nl = np.asarray(t_nl_jax)
    z_nl = np.asarray(z_full_jax)
    nu_nl = np.hstack([np.interp(t_nl, t_ref, nu_ref[:, i]).reshape((-1, 1)) for i in range(nu_ref.shape[1])])

    return t_nl, z_nl, nu_nl

def compile_tau_propagator(problem, method, n_dense=1000):
    fcn    = problem.constraints.get(type="dynamics")[0].fcn
    N      = method.index_map.N.time_grid
    params = tools.recursive_to_dict(problem.params)
    tau    = jnp.linspace(0.0, 1.0, N)
    tau_d  = jnp.linspace(0.0, 1.0, n_dense)

    @jax.jit
    def solve(z0, nu):
        def rhs(t, z, _):
            k = jnp.clip(jnp.searchsorted(tau, t, side='right') - 1, 0, N - 2)
            a = (tau[k+1] - t) / (tau[k+1] - tau[k])
            return fcn(t, z, a * nu[k] + (1 - a) * nu[k+1], params)
        return diffrax.diffeqsolve(
            diffrax.ODETerm(rhs), diffrax.Dopri5(), 0.0, 1.0, 1e-4, z0,
            stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
            saveat=diffrax.SaveAt(ts=tau_d), max_steps=100000,
        ).ys

    method.propagate_tau_jit = solve
    method.tau_nodes = tau
    method.tau_dense = tau_d


def propagate_tau(z0, nu_ref, problem, method):
    idx  = problem.index_map.indices
    z_nl = np.asarray(method.propagate_tau_jit(jnp.array(z0), jnp.array(nu_ref)))
    tau_n, tau_d = np.asarray(method.tau_nodes), np.asarray(method.tau_dense)
    u_nl = np.column_stack([np.interp(tau_d, tau_n, nu_ref[:, c]) for c in idx.nu.control])
    return z_nl[:, idx.z.time].reshape(-1), z_nl[:, idx.z.state], u_nl


def propagate_scipy_rk45(z0, nu_ref, t_ref, t_nl, problem, method, dynamics=None):

    dynamics = problem.constraints.get(type="dynamics")[0].fcn if dynamics is None else dynamics
    nu_interp = interp1d(t_ref, nu_ref, axis=0, fill_value="extrapolate")
    params = tools.recursive_to_dict(problem.params)

    def ode_func(t, z):
        nu = nu_interp(t)
        return np.array(_call_dynamics(dynamics, t, z, nu, params))

    sol = solve_ivp(ode_func, [t_nl[0], t_nl[-1]], z0, t_eval=t_nl, method='RK45', rtol=1e-6, atol=1e-8)

    t_out = sol.t
    z_out = sol.y.T
    nu_out = nu_interp(t_out)

    return t_out, z_out, nu_out


