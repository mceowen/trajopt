import numpy as np
from scipy.integrate import solve_ivp
import trajopt.utils.tools as tools
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


'''
outline of plt_data structure
scenario_data = {
    "method1": {
        "mc_data": [{"iters": {}, "params": {}}, {"iters": {}, "params": {}}, ...]
    },

    "method2": {
        "run_data": []
    }, 
}
'''

def rk4_propagate_jax_dense(dynamics, z0, nu_ref, t_ref, t_dense, problem):

    z0_jax = jnp.array(z0)
    nu_ref_jax = jnp.array(nu_ref)
    t_ref_jax = jnp.array(t_ref)
    t_dense_jax = jnp.array(t_dense)
    
    N_ref = len(t_ref_jax)
    N_dense = len(t_dense_jax)
    n = len(z0_jax)
    
    def rk4_step(zi, ti, dt, ui, ui_next):
        k1 = dynamics(ti, zi, ui, problem)
        u2 = 0.5 * ui + 0.5 * ui_next
        k2 = dynamics(ti + 0.5*dt, zi + 0.5*dt*k1, u2, problem)
        k3 = dynamics(ti + 0.5*dt, zi + 0.5*dt*k2, u2, problem)
        k4 = dynamics(ti + dt, zi + dt*k3, ui_next, problem)
        zi_next = zi + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return zi_next
    
    def interp_control_single(t, t_ref, nu_ref):
        """First-order hold interpolation of controls at a single time point."""
        idx = jnp.searchsorted(t_ref, t, side='right') - 1
        idx = jnp.clip(idx, 0, N_ref - 2)
        
        t0 = t_ref[idx]
        t1 = t_ref[idx + 1]
        u0 = nu_ref[idx]
        u1 = nu_ref[idx + 1]
        
        alpha = (t - t0) / (t1 - t0 + 1e-10)
        u = u0 + alpha * (u1 - u0)
        return u
    
    interp_control_vec = jax.vmap(interp_control_single, in_axes=(0, None, None))
    u_dense = interp_control_vec(t_dense_jax, t_ref_jax, nu_ref_jax)
    
    def scan_fn(zi, i):
        t = t_dense_jax[i]
        u = u_dense[i]
        dt = t_dense_jax[i+1] - t_dense_jax[i]
        u_next = u_dense[i+1]
        
        zi_next = rk4_step(zi, t, dt, u, u_next)
        return zi_next, zi_next
    
    def scan_wrapper(z0, xs):
        _, z_propagated = jax.lax.scan(scan_fn, z0, xs)
        return z_propagated
    
    scan_jit = jax.jit(scan_wrapper)
    z_propagated = scan_jit(z0_jax, jnp.arange(N_dense - 1))
    
    z_jax = jnp.vstack([z0_jax[None, :], z_propagated])
    z_numpy = np.array(z_jax)
    
    return z_numpy

def perform_default_analysis(problem):

    iter_data = problem.method.subprob.iter_data
    n = problem.model.n
    m = problem.model.m
    N = problem.method.N
    nondim = problem.method.nondim

    mission_params_exclude_list = ['_nonlinear_aero', 'costs', 'custom_modules', 'mission_module', '_get_cost_cnstr_nondim', '_set_custom_params', '_custom_constraints', '_custom_cost', 'problem'] 
    model_params_list = ['constraint_config_list', 'flags', 'm', 'n', 'name', 'nz', 'obs', 'u_types', 'z_types']
    method_params_list = ['N', 'N_dens', 'Npm', 'T_init', 'T_max', 'T_min', 'Ts_init', 'conv', 'conv_data', 'cost_init', 
                          'dT_max', 'ddt_max', 'dt_init', 'dt_init', 'dt_max', 'dt_min', 'flags', 'line_guess_u_init',
                          'name', 'n_minus', 'n_plus', 'nl_guess_u_start', 'nl_guess_u_stop', 'solver_opts',
                          'nondim', 't_init', 'nu_init', 'weights', 'z_ind', 'z_init']

    mission_params = tools.extract_attributes_exclude(problem.mission, exclude=mission_params_exclude_list)
    model_params   = tools.extract_attributes(problem.model, model_params_list)
    method_params  = tools.extract_attributes(problem.method, method_params_list)

    model_params['n'] = n
    model_params['m'] = m
    
    method_params['N'] = N
    method_params['nondim'] = nondim

    params_dict = {
        'mission': mission_params,
        'model': model_params,
        'method': method_params
    }

    odesettings = {"atol": 1e-12, "rtol": 1e-12}
    N_dense = 5 * N

    for data in iter_data:
        
        # get reference trajectory for this iteration (in nondimensional coordinates)
        t_ref = np.asarray(data['t_ref'])
        z_ref = np.asarray(data['z_ref'])
        nu_ref = np.asarray(data['us_ref'])
        
        # create dense time grid for this iteration based on its reference trajectory time span
        t_dense = np.linspace(t_ref[0], t_ref[-1], N_dense)
        
        # create dense control interpolation for this iteration
        nu_ref_dense = np.hstack([np.interp(t_dense, t_ref, nu_ref[:, i]).reshape((-1, 1)) for i in range(m)])
        u_ref_dense = nu_ref_dense @ nondim['M']['ctrl']['nd2d']
        
        # TODO: need to move this to an integrator module
        # choose integrator based on jax_dyn flag
        use_jax = problem.method.flags.get("jax_dyn", 0)
        
        z_ref_np = np.asarray(z_ref)
        
        if use_jax:
            # use JAX-based RK4 propagation
            z_nl = rk4_propagate_jax_dense(
                problem.model._dynamics,
                z_ref_np[0, :n],
                nu_ref,
                t_ref,
                t_dense,
                problem
            )
            data['t_nl'] = t_dense * nondim['nt']
            data['z_nl'] = z_nl @ nondim['M']['state']['nd2d']
            data['u_nl'] = u_ref_dense
        else:
            # use scipy solve_ivp
            def FOH_dynamics(t, z, nu_ref, t_ref):
                """First-order hold dynamics for RK45 integration."""
                # Interpolate control at time t (each control dimension separately)
                u_t = np.array([np.interp(t, t_ref, nu_ref[:, i]) for i in range(m)])
                # Call model dynamics
                return problem.model.dynamics(t, z, u_t)
            
            sol = solve_ivp(
                FOH_dynamics,
                [t_ref[0], t_ref[-1]],
                z_ref_np[0, :n],
                args=(nu_ref, t_ref),
                t_eval=t_dense,
                method='RK45',
                **odesettings
            )
            
            data['t_nl'] = t_dense * nondim['nt']
            data['z_nl'] = sol.y.T @ nondim['M']['state']['nd2d']
            data['u_nl'] = u_ref_dense

        data['t_ref'] = data['t_ref'] * nondim['nt']
        data['z_ref'] = data['z_ref'][:, :n] @ nondim['M']['state']['nd2d']
        data['u_ref'] = data['us_ref'] @ nondim['M']['ctrl']['nd2d']

        if 'ts' in data:
            data['t'] = data['ts'] * nondim['nt']
            data['z'] = data['zs'][:, :n] @ nondim['M']['state']['nd2d']
            data['u'] = data['us'] @ nondim['M']['ctrl']['nd2d']

    return {'iters': iter_data, 'params': params_dict}