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

    def propagate(z0, nu_ref, t_ref, t_nl):

        dt = t_nl[1] - t_nl[0]

        def rk4_step_partial(zi, ti):
            zi_next = rk4_step_jit(zi, ti, dt, nu_ref, t_ref)
            return zi_next, zi_next

        zf, zs = jax.lax.scan(rk4_step_partial, z0, t_nl[:-1])
        
        z_full = jnp.vstack([z0[None, :], zs])

        return z_full

    method.propagate_rk4_dense_jit = jax.jit(propagate)

def propagate_rk4_dense(z0, nu_ref, t_ref, t_nl, method):

    z0_jax = jnp.array(z0)
    nu_ref_jax = jnp.array(nu_ref)
    t_ref_jax = jnp.array(t_ref)
    t_nl_jax = jnp.array(t_nl)

    z_full_jax = method.propagate_rk4_dense_jit(z0_jax, nu_ref_jax, t_ref_jax, t_nl_jax)

    return np.asarray(z_full_jax)

def nonlinear_propagation(t_opt, z_opt, nu_opt, problem, method):
    
    n = problem.n
    m = problem.m
    N = method.N
    
    odesettings = {"atol": 1e-12, "rtol": 1e-12}
    N_dense =  100 * N

    # create dense time grid for this iteration based on its reference trajectory time span
    t_nl = np.linspace(t_opt[0], t_opt[-1], N_dense)
    
    # create dense control interpolation for this iteration
    nu_nl = np.hstack([np.interp(t_nl, t_opt, nu_opt[:, i]).reshape((-1, 1)) for i in range(m)])

    # choose integrator based on jax_dyn flag
    use_jax = method.flags.get("jax_dyn", 0)
    
    if use_jax:
        # use JAX-based RK4 propagation
        z_nl = propagate_rk4_dense(z_opt[0, :n], nu_opt, t_opt, t_nl, method)
    else:
        # use scipy solve_ivp
        def FOH_dynamics(t, z, nu_opt, t_opt):
            """First-order hold dynamics for RK45 integration."""
            # Interpolate control at time t (each control dimension separately)
            u_t = np.array([np.interp(t, t_opt, nu_opt[:, i]) for i in range(m)])
            # Call model dynamics
            return  problem.dynamics(t, z, u_t)
        
        sol = solve_ivp(FOH_dynamics,[t_opt[0], t_opt[-1]],z_opt[0, :n],args=(nu_opt, t_opt),t_eval=t_nl,method='RK45',**odesettings)
        z_nl = sol.y.T
    
    return t_nl, z_nl, nu_nl