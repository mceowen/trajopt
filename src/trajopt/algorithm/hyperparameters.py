import numpy as np

def autotune1(O, problem, iter_num, vb_path, vb_nfz, vb_aux, vb_dyn, vb_term, dual_path, dual_nfz, dual_aux, dual_dyn, dual_term):

    # Extract hyperparameters
    if problem['params']['bools']['stepsize_auto_dual']:
        beta = 1 / iter_num
        gamma = 1 / iter_num
    else:
        beta = problem['params']['weights']['beta']
        gamma = problem['params']['weights']['gamma']

    # Convert value buffers to numpy arrays if they are not already
    vb_path = np.array(vb_path)
    vb_nfz = np.array(vb_nfz)
    vb_aux = np.array(vb_aux)
    vb_dyn = np.array(vb_dyn)
    vb_term = np.array(vb_term)

    eps_feas_path = problem['params']['conv']['eps_path']
    eps_feas_nfz = problem['params']['conv']['eps_nfz']
    eps_feas_aux = problem['params']['conv']['eps_aux']
    eps_feas_term = problem['params']['conv']['eps_term']
    eps_feas_dyn = problem['params']['conv']['eps_dyn']

    # Inequality
    dual_path_plus = np.maximum(0, gamma * vb_path + dual_path)
    dual_nfz_plus = np.maximum(0, gamma * vb_nfz + dual_nfz)
    dual_aux_plus = np.maximum(0, gamma * vb_aux + dual_aux)
    dmu_ineq = np.concatenate([dual_path_plus, dual_nfz_plus, dual_aux_plus]) - np.concatenate([dual_path, dual_nfz, dual_aux])

    # Saturate
    n_path = len(vb_path)
    n_nfz = len(vb_nfz)
    n_aux = len(vb_aux)

    if n_path > 0:
        dual_path_plus[vb_path <= eps_feas_path] = dual_path[vb_path <= eps_feas_path]
    if n_nfz > 0:
        dual_nfz_plus[vb_nfz <= eps_feas_nfz] = dual_nfz[vb_nfz <= eps_feas_nfz]
    if n_aux > 0:
        dual_aux_plus[vb_aux <= eps_feas_aux] = dual_aux[vb_aux <= eps_feas_aux]

    O['weights']['dual_path'] = dual_path_plus
    O['weights']['dual_nfz'] = dual_nfz_plus
    O['weights']['dual_aux'] = dual_aux_plus
    O['weights']['data']['dmu_ineq'] = dmu_ineq

    # Equality
    dual_dyn_plus = beta * vb_dyn + dual_dyn
    dual_term_plus = beta * vb_term + dual_term
    dmu_eq = dual_term_plus - dual_term

    # Saturate
    dual_dyn_plus[np.abs(vb_dyn) <= eps_feas_dyn] = dual_dyn[np.abs(vb_dyn) <= eps_feas_dyn]
    dual_term_plus[np.abs(vb_term) <= eps_feas_term] = dual_term[np.abs(vb_term) <= eps_feas_term]

    O['weights']['dual_dyn'] = dual_dyn_plus
    O['weights']['dual_term'] = dual_term_plus
    O['weights']['data']['dmu_eq'] = dmu_eq

    return O

def autotune2(O, problem, N, vb_path, vb_nfz, vb_aux, vb_dyn, vb_term, W_path, W_nfz, W_aux, W_dyn, W_term):

    # Extract parameters for autotuning
    eps_feas_path = problem['params']['conv']['eps_path']
    eps_feas_nfz = problem['params']['conv']['eps_nfz']
    eps_feas_aux = problem['params']['conv']['eps_aux']
    eps_feas_term = problem['params']['conv']['eps_term']
    eps_feas_dyn = problem['params']['conv']['eps_dyn']

    eps_nonzero2 = problem['params']['weights']['eps_nonzero2']
    flag_Wmemory = problem['params']['bools']['flag_Wauto_memory']

    buff_dyn = problem['params']['bools']['buff_dyn']

    path_idx = problem['params']['path_idx']
    nfz_idx = problem['params']['nfz_idx']
    aux_idx = problem['params']['aux_idx']
    dual_ineq = []
    dual_path_buff = []
    dual_nfz_buff = []
    dual_aux_buff = []
    dual_dyn_buff = []
    Wh_path = []
    Wh_nfz = []
    Wh_aux = []
    Wh_dyn = []
    Wh_term = []

    # Autotune matrices via dual variables and feasibility tolerance
    for k in range(N):
        dual_path_buff.append(np.diag(W_path[:, k]) @ vb_path[:, k].flatten())
        dual_nfz_buff.append(np.diag(W_nfz[:, k]) @ vb_nfz[:, k].flatten())
        dual_aux_buff.append(np.diag(W_aux[:, k]) @ vb_aux[:, k].flatten())

        if problem['params']['n_ineq'] > 0:
            if problem['params']['n_path'] > 0:
                Wh_path.append(np.abs(dual_path_buff[-1] / eps_feas_path))
            if problem['params']['n_nfz'] > 0:
                Wh_nfz.append(np.abs(dual_nfz_buff[-1] / eps_feas_nfz))
            if problem['params']['n_aux'] > 0:
                Wh_aux.append(np.abs(dual_aux_buff[-1] / eps_feas_aux))
        else:
            Wh_path.append(np.abs(dual_path_buff[-1]))
            Wh_nfz.append(np.abs(dual_nfz_buff[-1]))
            Wh_aux.append(np.abs(dual_aux_buff[-1]))

        if k < N - 1:
            dual_dyn_buff.append(np.diag(W_dyn[:, k].flatten()) @ vb_dyn[:, k])
            if buff_dyn:
                Wh_dyn.append(np.sum(np.abs(dual_dyn_buff[-1]) / eps_feas_dyn))
            else:
                Wh_dyn.append(np.sum(np.abs(dual_dyn_buff[-1])))

    if (problem['params']['n_term'] + problem['params']['n_term_ineq']) > 0:
        dual_term_buff = np.diag(W_term.flatten()) @ vb_term
        Wh_term = np.abs(dual_term_buff / eps_feas_term)

    # Extract field names and create buffer nametags
    W_fn = [key for key in problem['params']['weights'].keys() if key.startswith('W')]
    nametags = [key.split('_')[1] for key in W_fn if key.startswith('W')]

    for i_field in nametags:
        W_field = f'W_{i_field}'
        Wh_field = f'Wh_{i_field}'
        vb_field = f'vb_{i_field}'
        eps_feas = f'eps_feas_{i_field}'
        Wconv_field = f'Wconv_{i_field}'

        if np.sum(problem['params']['weights'][W_field]) == 0:
            O['weights'][W_field] = eval(W_field)
        else:
            if flag_Wmemory == 0:
                # Remove nonzero elements from new candidate weight (derived from dual)
                exec(f"{Wh_field}[{Wh_field} <= eps_nonzero2] = eps_nonzero2")
            elif flag_Wmemory == 1:
                # Stop updating weight after desired threshold
                exec(f"eps_feas = {eps_feas}")
                exec(f"Wconv_field = problem['params']['conv'][{Wconv_field}]")
                exec(f"idx_feas_thresh = (Wconv_field @ value({vb_field}) <= eps_feas)")
                exec(f"{Wh_field}[idx_feas_thresh] = {W_field}[idx_feas_thresh]")
            elif flag_Wmemory == 2:
                exec(f"eps_feas = {eps_feas}")
                exec(f"Wconv_field = problem['params']['conv'][{Wconv_field}]")
                exec(f"idx_feas_thresh = (Wconv_field @ value({vb_field}) <= eps_feas)")
                exec(f"{Wh_field}[idx_feas_thresh] = np.minimum(eps_nonzero2, {W_field}[idx_feas_thresh])")

            # Create updated weight
            O['weights'][W_field] = eval(Wh_field)

    # TODO - clean me
    O['weights']['data']['eps_feas'] = eps_feas_path

    # CHECKS
    O['weights']['data'] = {}

    O['weights']['data']['term'] = {
        'Wxq': np.diag(W_term.flatten()) @ vb_term,
        'dual': dual_term_buff
    }

    for k in range(N):
        O['weights']['data']['path'] = {
            'Wxq': np.diag(W_path[:, k].flatten()) @ vb_path[:, k],
            'dual': dual_path_buff[k]
        }
        O['weights']['data']['nfz'] = {
            'Wxq': np.diag(W_nfz[:, k].flatten()) @ vb_nfz[:, k],
            'dual': dual_nfz_buff[k]
        }
        O['weights']['data']['aux'] = {
            'Wxq': np.diag(W_aux[:, k].flatten()) @ vb_aux[:, k],
            'dual': dual_aux_buff[k]
        }
        # if k < N - 1:
        #     O['weights']['data']['dyn'] = {
        #         'Wxq': np.diag(W_dyn[:, k].flatten()) @ vb_dyn[:, k],
        #         'dual': dual_dyn_buff[k]
        #     }

    O['weights']['data']['term']['delta'] = O['weights']['data']['term']['Wxq'] - O['weights']['data']['term']['dual']
    O['weights']['data']['path']['delta'] = O['weights']['data']['path']['Wxq'] - O['weights']['data']['path']['dual']
    O['weights']['data']['nfz']['delta'] = O['weights']['data']['nfz']['Wxq'] - O['weights']['data']['nfz']['dual']
    O['weights']['data']['aux']['delta'] = O['weights']['data']['aux']['Wxq'] - O['weights']['data']['aux']['dual']
    # O['weights']['data']['dyn']['delta'] = O['weights']['data']['dyn']['Wxq'] - O['weights']['data']['dyn']['dual']

    return O


def autotune3(O, problem, iter_num, vb_path, vb_nfz, vb_aux, vb_dyn, vb_term, dual_path, dual_nfz, dual_aux, dual_dyn, dual_term):
    O = autotune1(O, problem, N, vb_path, vb_nfz, vb_aux, vb_dyn, vb_term, W_path, W_nfz, W_aux, W_dyn, W_term)
    O = autotune2(O, problem, N, vb_path, vb_nfz, vb_aux, vb_dyn, vb_term, W_path, W_nfz, W_aux, W_dyn, W_term)

    return O


# Example usage
if __name__ == "__main__":
    # Define dummy data for testing
    O = {
        'weights': {
            'dual_path': None,
            'dual_nfz': None,
            'dual_aux': None,
            'dual_dyn': None,
            'dual_term': None,
            'data': {
                'dmu_ineq': None,
                'dmu_eq': None
            }
        }
    }

    problem = {
        'params': {
            'bools': {
                'stepsize_auto_dual': True
            },
            'weights': {
                'beta': 0.1,
                'gamma': 0.1
            },
            'conv': {
                'eps_path': 0.01,
                'eps_nfz': 0.01,
                'eps_aux': 0.01,
                'eps_term': 0.01,
                'eps_dyn': 0.01
            }
        }
    }

    iter_num = 10
    vb_path = np.array([0.02, 0.03])
    vb_nfz = np.array([0.02])
    vb_aux = np.array([0.02])
    vb_dyn = np.array([0.02])
    vb_term = np.array([0.02])

    dual_path = np.array([0.01, 0.01])
    dual_nfz = np.array([0.01])
    dual_aux = np.array([0.01])
    dual_dyn = np.array([0.01])
    dual_term = np.array([0.01])

    updated_O = autotune1(O, problem, iter_num, vb_path, vb_nfz, vb_aux, vb_dyn, vb_term, dual_path, dual_nfz, dual_aux, dual_dyn, dual_term)
    print(updated_O)


    # Define dummy data for testing
    O = {
        'weights': {
            'dual_path': None,
            'dual_nfz': None,
            'dual_aux': None,
            'dual_dyn': None,
            'dual_term': None,
            'data': {
                'dmu_ineq': None,
                'dmu_eq': None
            }
        }
    }

    problem = {
        'params': {
            'bools': {
                'flag_Wauto_memory': 0,
                'buff_dyn': True
            },
            'weights': {
                'eps_nonzero2': 1e-6,
                'W_path': np.zeros((2, 3)),
                'W_nfz': np.zeros((1, 3)),
                'W_aux': np.zeros((1, 3)),
                'W_dyn': np.zeros((1, 2)),
                'W_term': np.zeros((1,))
            },
            'conv': {
                'eps_path': 0.01,
                'eps_nfz': 0.01,
                'eps_aux': 0.01,
                'eps_term': 0.01,
                'eps_dyn': 0.01
            },
            'path_idx': [0, 1],
            'nfz_idx': [0],
            'aux_idx': [0]
        }
    }

    N = 3
    vb_path = np.random.rand(2, 3)
    vb_nfz = np.random.rand(1, 3)
    vb_aux = np.random.rand(1, 3)
    vb_dyn = np.random.rand(1, 2)
    vb_term = np.random.rand(1,)

    W_path = np.random.rand(2, 3)
    W_nfz = np.random.rand(1, 3)
    W_aux = np.random.rand(1, 3)
    W_dyn = np.random.rand(1, 2)
    W_term = np.random.rand(1,)

    updated_O = autotune2(O, problem, N, vb_path, vb_nfz, vb_aux, vb_dyn, vb_term, W_path, W_nfz, W_aux, W_dyn, W_term)
    print(updated_O)

    updated_O = autotune3(O, problem, N, vb_path, vb_nfz, vb_aux, vb_dyn, vb_term, W_path, W_nfz, W_aux, W_dyn, W_term)
    print(updated_O)