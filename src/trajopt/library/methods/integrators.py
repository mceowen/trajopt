import numpy as np
import jax
import jax.numpy as jnp
import trajopt.utils.tools as tools

def compile_dense_jax_propagator(problem, method, params):

    dynamics = problem.constraints.get(type="dynamics")[0].fcn

    def rk4_step(zi, ti, dt, nu_ref, t_ref, params):
        k1 = z_dot(zi, ti, nu_ref, t_ref, params)
        k2 = z_dot(zi + 0.5*dt*k1, ti + 0.5*dt, nu_ref, t_ref, params)
        k3 = z_dot(zi + 0.5*dt*k2, ti + 0.5*dt, nu_ref, t_ref, params)
        k4 = z_dot(zi + dt*k3, ti + dt, nu_ref, t_ref, params)
        return zi + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def z_dot(z, t, nu_ref, t_ref, params):
        k = jnp.searchsorted(t_ref, t, side='right') - 1
        k = jnp.clip(k, 0, len(t_ref) - 2)

        nu_ref_k = nu_ref[k]
        nu_ref_kp = nu_ref[k+1]
        t_ref_k = t_ref[k]
        t_ref_kp = t_ref[k+1]

        # FOH interpolation of controls
        a = 1 - (t - t_ref_k) / (t_ref_kp - t_ref_k)
        b = (t - t_ref_k) / (t_ref_kp - t_ref_k)
        nu = a * nu_ref_k + b * nu_ref_kp

        return dynamics(t, z, nu, params)

    rk4_step_jit = rk4_step

    def propagate(z0, nu_ref, t_ref, t_nl, params):
        dt = t_nl[1] - t_nl[0]

        def rk4_step_partial(zi, ti):
            zi_next = rk4_step_jit(zi, ti, dt, nu_ref, t_ref, params)
            return zi_next, zi_next

        zf, zs = jax.lax.scan(rk4_step_partial, z0, t_nl[:-1])

        z_full = jnp.vstack([z0[None, :], zs])

        return z_full

    method.propagate_rk4_dense_jit = jax.jit(propagate)

def propagate_jax_rk4_dense(z0, nu_ref, t_ref, t_nl, problem, method):

    z0_jax = jnp.array(z0)
    nu_ref_jax = jnp.array(nu_ref)
    t_ref_jax = jnp.array(t_ref)
    t_nl_jax = jnp.array(t_nl)
    
    params = problem.params
    params_jax = tools.recursive_to_dict(params)
    
    z_full_jax = method.propagate_rk4_dense_jit(z0_jax, nu_ref_jax, t_ref_jax, t_nl_jax, params_jax)

    t_nl = np.asarray(t_nl_jax)
    z_nl = np.asarray(z_full_jax)
    nu_nl = np.hstack([np.interp(t_nl, t_ref, nu_ref[:, i]).reshape((-1, 1)) for i in range(problem.index_map.n['control'])])

    return t_nl, z_nl, nu_nl

def propagate_scipy_rk45(z0, nu_ref, t_ref, t_nl, problem, method):

    pass 
