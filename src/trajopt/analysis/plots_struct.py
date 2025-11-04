import numpy as np
from scipy.integrate import solve_ivp
import trajopt.core.modules.methods.scaling as scaling

def initialize_plot_struct():

    plt = {}
    plt['scenario1'] = {}
    plt['scenario1']['method1'] = {}
    plt['scenario1']['method1']['params'] = {}
    plt['scenario1']['method1']['runs'] = []
    
    return plt

# iters: traj opt nonlin prop ref * constraints (alg params inside)

def perform_default_analysis(problem):

    # extract inputs and outputs form problem struct
    # excluding cvxpy problem and dynamics matrices
    exclude = {'subprob', 'Ak', 'Bk', 'Bkp', 'Sk'}

    I0 = {key: val for key, val in problem['I'][0].items()}
    O = [{key: val for key, val in iter.items() if key not in exclude} for iter in problem['O']]

    O.insert(0, I0.copy())
    params = problem['params']

    n = params['n']

    iters = [{} for _ in range(len(O))]

    # extract final optimized trajectory
    ts_opt = O[-1]['ts']
    zs_opt = O[-1]['zs']
    us_opt = O[-1]['us']

    N_dense = 20 * params['N']
    ts_dense = np.linspace(ts_opt[0], ts_opt[-1], N_dense)
    t_dense = ts_dense * params['nondim']['nt']

    us_opt_dense = np.hstack([np.interp(ts_dense, ts_opt, us_opt[:, i]).reshape((-1, 1)) for i in range(params['m'])])
    u_opt_dense = us_opt_dense @ params['nondim']['M']['ctrl']['nd2d']

    # run nonlinear propagation on dimensionalized variables
    system_dynamics = problem['xdot'] # function of (ts, zs, us, t_vec)

    def dyn_wrapped(ts, zs):
        return system_dynamics(ts, zs, us_opt, t_vec=ts_opt)

    # add dimensionalized variables for each iteration
    for i in range(len(iters)):

        sol = solve_ivp(dyn_wrapped,
                t_span=(ts_opt[0],ts_opt[-1]),
                y0=zs_opt[0, :n],
                t_eval=ts_dense,
                method='RK45', 
                rtol=1e-12,
                atol=1e-12
                )

        # store nonlinear propagation into analysis struct
        iters[i]['t_nl'] = t_dense
        iters[i]['z_nl'] = sol.y.T @ params['nondim']['M']['state']['nd2d']
        iters[i]['u_nl'] = u_opt_dense

        # add dimensional reference trejectories
        ts_ref = O[i]['ts_ref']
        zs_ref = O[i]['zs_ref']
        us_ref = O[i]['us_ref']

        iters[i]['t_ref'] = ts_ref * params['nondim']['nt']
        iters[i]['z_ref'] = zs_ref[:, :n] @ params['nondim']['M']['state']['nd2d']
        iters[i]['u_ref'] = us_ref @ params['nondim']['M']['ctrl']['nd2d']

         # add dimensional optimized trajectories
        if i > 0:
            ts = O[i]['ts']
            zs = O[i]['zs'][:, :n]
            us = O[i]['us']

            iters[i]['t'] = ts * params['nondim']['nt']
            iters[i]['z'] = zs[:, :n] @ params['nondim']['M']['state']['nd2d']
            iters[i]['u'] = us @ params['nondim']['M']['ctrl']['nd2d']
            
            iters[i]['weights'] = O[i]['weights']

            remove = ['soln']
            iters[i]['conv_data'] = {k: v for k, v, in O[i]['conv_data'].items() if k not in remove}

    run_data = {'iters': iters}
    
    return run_data

if __name__ == "__main__":
    pass