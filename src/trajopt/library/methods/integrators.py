import numpy as np
import jax
import jax.numpy as jnp
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

    rk4_step_jit = jax.jit(rk4_step)

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


