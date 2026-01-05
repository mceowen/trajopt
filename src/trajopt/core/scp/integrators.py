import numpy as np
import jax
import jax.numpy as jnp

def jit_rk4_jax_dense(problem, method):

    dynamics = problem.constraints.get('name', 'dynamics')[0].fcn
    
    def rk4_step(zi, ti, dt, nu_ref, t_ref):
        k1 = z_dot(zi, ti, nu_ref, t_ref)
        k2 = z_dot(zi + 0.5*dt*k1, ti + 0.5*dt, nu_ref, t_ref)
        k3 = z_dot(zi + 0.5*dt*k2, ti + 0.5*dt, nu_ref, t_ref)
        k4 = z_dot(zi + dt*k3, ti + dt, nu_ref, t_ref)
        return zi + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def z_dot(z, t, nu_ref, t_ref):

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

        return dynamics(t, z, nu)

    rk4_step_jit = jax.jit(rk4_step)

    def propagate(z0, nu_ref, t_ref, t_dense):

        dt = t_dense[1] - t_dense[0]

        def rk4_step_partial(zi, ti):
            zi_next = rk4_step_jit(zi, ti, dt, nu_ref, t_ref)
            return zi_next, zi_next

        zf, zs = jax.lax.scan(rk4_step_partial, z0, t_dense[:-1])
        
        z_full = jnp.vstack([z0[None, :], zs])

        return z_full

    method.propagate_rk4_dense_jit = jax.jit(propagate)

def propagate_rk4_dense(z0, nu_ref, t_ref, t_dense, method):

    z0_jax = jnp.array(z0)
    nu_ref_jax = jnp.array(nu_ref)
    t_ref_jax = jnp.array(t_ref)
    t_dense_jax = jnp.array(t_dense)

    z_full_jax = method.propagate_rk4_dense_jit(z0_jax, nu_ref_jax, t_ref_jax, t_dense_jax)

    return np.asarray(z_full_jax)