import numpy as np

import trajopt.utils.tools as tools

def set_convergence_tolerance(params):
    
    # STATE CONVERGENCE
    n                   = params['nz']
    ctcs_mult_state     = params['conv']['setup']['ctcs_mult_state']
    ctcs_mult_cnst      = params['conv']['setup']['ctcs_mult_cnst']

    if len(params['conv']['setup']['eps_state']) == 1:
        eps_state       = params['conv']['setup']['eps_state'] * np.ones(n)
        M_state_d2nd    = np.eye(n)
    else:
        eps_state       = params['conv']['setup']['eps_state']
        M_state_d2nd    = params['nondim']['M']['state']['d2nd']

    if params['bools']['ctcs'] and params['n_ineq'] > 0:
        eps_state = np.concatenate([
            ctcs_mult_state * eps_state,
            ctcs_mult_cnst  * params['conv']['setup']['eps_path'],
            ctcs_mult_cnst  * params['conv']['setup']['eps_nfz'],
            ctcs_mult_cnst  * params['conv']['setup']['eps_aux']
        ])
        M_state_d2nd = np.diag(np.concatenate([
            np.diag(M_state_d2nd),
            np.diag(params['nondim']['M']['cnst']['d2nd'])
        ]))
    
    eps_state_nd                        = M_state_d2nd @ eps_state
    eps_min_state                       = float(np.min(eps_state_nd))
    
    Wconv_state                         = np.diag(eps_min_state / eps_state_nd)
    
    params['conv']['eps_state']         = eps_min_state 
    params['conv']['Wconv_state']       = Wconv_state
    params['conv']['Wconv_state_vec']   = np.diag(Wconv_state).copy() # .copy() makes sure it is contiguous

    # COST CONVERGENCE
    eps_cost                            = params['conv']['setup']['eps_cost']
    M_cost_d2nd                         = params['nondim']['M']['cost']['d2nd']
    
    params['conv']['eps_cost']          = M_cost_d2nd * eps_cost

    # NONCONVEX PATH CONSTRAINT CONVERGENCE
    n_path = params['n_path']

    if n_path > 0:
        if len(params['conv']['setup']['eps_path']) == 1:
            eps_path        = params['conv']['setup']['eps_path'] * np.ones(n_path)
            M_path_d2nd     = np.eye(n_path)
            eps_path_nd     = M_path_d2nd @ eps_path
            eps_min_path    = float(np.min(eps_path_nd))
        else:
            eps_path        = params['conv']['setup']['eps_path']
            M_path_d2nd     = params['nondim']['M_path_d2nd']
            eps_path_nd     = M_path_d2nd @ eps_path
            eps_min_path    = float(np.min(eps_path_nd))
    else:
        eps_path            = 0.
        M_path_d2nd         = np.zeros((1, n_path))
        eps_path_nd         = M_path_d2nd * eps_path
        eps_min_path        = 0.
    
    Wconv_path                          = np.diag(eps_min_path / eps_path_nd)
    
    params['conv']['eps_path']          = eps_min_path 
    params['conv']['Wconv_path']        = Wconv_path
    params['conv']['Wconv_path_vec']    = np.diag(Wconv_path).copy()

    # NONCONVEX NFZ CONSTRAINT CONVERGENCE
    n_nfz = params['n_nfz']

    if n_nfz > 0:
        if len(params['conv']['setup']['eps_nfz']) == 1:
            eps_nfz     = params['conv']['setup']['eps_nfz'] * np.ones(n_nfz)
            M_nfz_d2nd  = np.eye(n_nfz)
            eps_nfz_nd  = M_nfz_d2nd @ eps_nfz
            eps_min_nfz = float(np.min(eps_nfz_nd))
        else:
            eps_nfz     = params['conv']['setup']['eps_nfz']
            M_nfz_d2nd  = params['nondim']['M']['nfz']['d2nd']
            eps_nfz_nd  = M_nfz_d2nd @ eps_nfz
            eps_min_nfz = float(np.min(eps_nfz_nd))
    else:
        eps_nfz         = 0.
        M_nfz_d2nd      = np.zeros((1, n_nfz))
        eps_nfz_nd      = M_nfz_d2nd * eps_nfz
        eps_min_nfz     = 0.

    Wconv_nfz = np.diag(eps_min_nfz / eps_nfz_nd)
    
    params['conv']['eps_nfz']       = eps_min_nfz 
    params['conv']['Wconv_nfz']     = Wconv_nfz
    params['conv']['Wconv_nfz_vec'] = np.diag(Wconv_nfz).copy()

    # NONCONVEX AUXILIARY CONSTRAINT CONVERGENCE
    n_aux = params['n_aux']

    if n_aux > 0:
        if len(params['conv']['setup']['eps_aux']) == 1:
            eps_aux     = params['conv']['setup']['eps_aux'] * np.ones(n_aux)
            M_aux_d2nd  = np.eye(n_aux)
        else:
            eps_aux     = params['conv']['setup']['eps_aux']
            M_aux_d2nd  = params['nondim']['M_aux_d2nd']
            
        eps_aux_nd      = M_aux_d2nd @ eps_aux
        eps_min_aux     = float(np.min(eps_aux_nd))
    else:
        eps_aux         = 0.
        M_aux_d2nd      = np.zeros((1, n_aux))
        eps_aux_nd      = M_aux_d2nd * eps_aux
        eps_min_aux     = 0.

    Wconv_aux                       = np.diag(eps_min_aux / eps_aux_nd)
    
    params['conv']['eps_aux']       = eps_min_aux 
    params['conv']['Wconv_aux']     = Wconv_aux
    params['conv']['Wconv_aux_vec'] = np.diag(Wconv_aux).copy()

    # TERMINAL CONSTRAINT CONVERGENCE
    n_term = params['n_term'] + params['n_term_ineq']

    if n_term > 0:
        if len(params['conv']['setup']['eps_term']) == 1 and (params['n_term'] + params['n_term_ineq']) != 1:
            eps_term    = params['conv']['setup']['eps_term'] * np.ones(n_term)
            M_term_d2nd = np.eye(n_term)
        else:
            eps_term    = params['conv']['setup']['eps_term']
            M_term_d2nd = params['nondim']['M']['term']['d2nd']
        eps_term_nd     = M_term_d2nd @ eps_term
        eps_min_term    = float(np.min(eps_term_nd))
    else:
        eps_term        = 0.
        M_term_d2nd     = np.zeros((1, n_term))
        eps_term_nd     = M_term_d2nd * eps_term
        eps_min_term    = 0.
    
    Wconv_term                          = np.diag(eps_min_term / eps_term_nd)
    

    params['conv']['eps_term']          = eps_min_term 
    params['conv']['Wconv_term']        = Wconv_term
    params['conv']['Wconv_term_vec']    = np.diag(Wconv_term).copy()

    # MULTIPLE SHOOTING DYNAMICS DEFECT CONVERGENCE
    if len(params['conv']['setup']['eps_defect']) == 1:
        eps_defect      = params['conv']['setup']['eps_defect'] * np.ones(n)
        M_defect_d2nd   = np.eye(n)
    else:
        eps_defect      = params['conv']['setup']['eps_defect']
        M_defect_d2nd   = params['nondim']['M']['state']['d2nd']

    eps_defect_nd       = M_defect_d2nd @ eps_defect
    eps_min_defect      = float(np.min(eps_defect_nd))
    
    Wconv_defect                        = np.diag(eps_min_defect / eps_defect_nd)
    
    params['conv']['eps_defect']        = eps_min_defect 
    params['conv']['Wconv_defect']      = Wconv_defect
    params['conv']['Wconv_defect_vec']  = np.diag(Wconv_defect).copy()

    # DYNAMICS CONVERGENCE
    n_dyn = params['n_dyn']

    # extract convergence tolerance with dimensional units and nondimensionalization factor
    if n_dyn > 0:
        eps_dyn     = params['conv']['setup']['eps_dyn']
        M_dyn_d2nd  = params['nondim']['M']['dyn']['d2nd']
    else:
        eps_dyn     = np.zeros((params['nz'], params['nz'])) # make sure we can mat-mult w/ M_dyn_d2nd and get a 1-by-nz
        M_dyn_d2nd  = np.zeros((1, params['nz']))

    if params['bools']['ctcs'] and params['n_ineq'] > 0:
        eps_dyn = np.concatenate([
            ctcs_mult_state * eps_dyn,
            ctcs_mult_cnst * params['conv']['setup']['eps_path'],
            ctcs_mult_cnst * params['conv']['setup']['eps_nfz'],
            ctcs_mult_cnst * params['conv']['setup']['eps_aux']
        ])
        M_dyn_d2nd = np.diag(np.concatenate([
            np.diag(M_dyn_d2nd),
            np.diag(params['nondim']['M']['cnst']['d2nd'])
        ]))
    
    eps_dyn_nd                      = M_dyn_d2nd @ eps_dyn
    eps_min_dyn                     = float(np.min(eps_dyn_nd))
    
    if np.all(eps_dyn_nd == 0):
        Wconv_dyn   = np.diag(eps_dyn_nd) # zero matrix
    else:
        Wconv_dyn   = np.diag(eps_min_dyn / eps_dyn_nd)
    
    params['conv']['eps_dyn']       = eps_min_dyn 
    params['conv']['Wconv_dyn']     = Wconv_dyn
    params['conv']['Wconv_dyn_vec'] = np.diag(Wconv_dyn).copy()

    return params

def check_convergence_tolerance(problem, subprob, O):
    """Check convergence using the new Subproblem object interface."""

    # --- Load convergence data
    conv_data = O['conv_data']
    soln      = conv_data['soln']

    # --- Extract dimensions
    n = subprob.nz
    N = subprob.N

    # --- Extract optimization variables
    dz       = O['dz_s']
    dcost    = O['cost'] - conv_data['cost_ref']
    defect   = conv_data['defect']
    vb_dyn   = conv_data['vb_dyn']
    vb_path  = conv_data['vb_path']
    vb_nfz   = conv_data['vb_nfz']
    vb_aux   = conv_data['vb_aux']
    vb_term  = conv_data['vb_term']

    # --- Extract convergence criteria
    conv = problem['params']['conv']
    eps_state  = conv['eps_state']
    eps_cost   = conv['eps_cost']
    eps_path   = conv['eps_path']
    eps_nfz    = conv['eps_nfz']
    eps_aux    = conv['eps_aux']
    eps_term   = conv['eps_term']
    eps_defect = conv['eps_defect']
    eps_dyn    = conv['eps_dyn']

    W_state  = conv['Wconv_state']
    W_path   = conv['Wconv_path']
    W_nfz    = conv['Wconv_nfz']
    W_aux    = conv['Wconv_aux']
    W_term   = conv['Wconv_term']
    W_dyn    = conv['Wconv_dyn']
    W_defect = conv['Wconv_defect']

    # --- Extract linear constraints
    cnst_path = O['cnst_path']
    params = problem['params']
    path_idx = np.array(params.get('path_idx', []))
    nfz_idx  = np.array(params.get('nfz_idx', []))
    aux_idx  = np.array(params.get('aux_idx', []))

    conv_path_nl = np.maximum(0.0, cnst_path[:, path_idx]) if path_idx.size else np.zeros((cnst_path.shape[0], 0))
    conv_nfz_nl  = np.maximum(0.0, cnst_path[:, nfz_idx])  if nfz_idx.size  else np.zeros((cnst_path.shape[0], 0))
    conv_aux_nl  = np.maximum(0.0, cnst_path[:, aux_idx])  if aux_idx.size  else np.zeros((cnst_path.shape[0], 0))

    # === Optimality ===
    dz_array = tools.safe_val(dz, rows=N, cols=n)
    chk_dz   = np.max([np.max(W_state @ np.abs(dz_k)) for dz_k in dz_array])
    chk_cost = np.abs(dcost)

    # === Feasibility: Virtual Buffer violations ===
    chk_vb_path = np.max([np.max(W_path @ vb_path[k].reshape(-1, 1)) for k in range(N)]) if vb_path.size else 0.0
    chk_vb_nfz  = np.max([np.max(W_nfz  @ vb_nfz[k].reshape(-1, 1))  for k in range(N)]) if vb_nfz.size  else 0.0
    chk_vb_aux  = np.max([np.max(W_aux  @ vb_aux[k].reshape(-1, 1))  for k in range(N)]) if vb_aux.size  else 0.0
    chk_vb_term = np.max(W_term * vb_term)
    chk_vb_dyn  = np.max([np.max(W_dyn @ vb_dyn[k].reshape(-1, 1)) for k in range(N - 1)]) if vb_dyn.size else 0.0
    chk_defect  = np.max([np.max(W_defect @ defect[k].reshape(-1, 1)) for k in range(N)]) if defect.size else 0.0

    # === Feasibility: Linearized constraint residuals ===
    chk_path_2 = np.max([np.max(W_path @ conv_path_nl[k].reshape(-1, 1)) for k in range(N)]) if conv_path_nl.size else 0.0
    chk_nfz_2  = np.max([np.max(W_nfz  @ conv_nfz_nl[k].reshape(-1, 1))  for k in range(N)]) if conv_nfz_nl.size  else 0.0
    chk_aux_2  = np.max([np.max(W_aux  @ conv_aux_nl[k].reshape(-1, 1))  for k in range(N)]) if conv_aux_nl.size  else 0.0

    # === Convergence mode selection
    ctcs      = params['bools']['ctcs']
    flag_conv = params['bools']['flag_conv']

    if ctcs:
        chk_feas_1 = np.array([chk_vb_term, chk_vb_dyn])
        chk_feas_2 = np.array([chk_vb_term, chk_defect])
        eps_feas_1 = np.array([eps_term, eps_dyn])
        eps_feas_2 = np.array([eps_term, eps_defect])
    else:
        chk_feas_1 = np.array([chk_vb_path, chk_vb_nfz, chk_vb_aux, chk_vb_term, chk_vb_dyn])
        chk_feas_2 = np.array([chk_path_2, chk_nfz_2, chk_aux_2, chk_vb_term])
        eps_feas_1 = np.array([eps_path, eps_nfz, eps_aux, eps_term, eps_dyn])
        eps_feas_2 = np.array([eps_path, eps_nfz, eps_aux, eps_term])

    if flag_conv == 0:
        chk_opt = np.array([chk_dz, chk_cost])
        eps_opt = np.array([eps_state, eps_cost])
    elif flag_conv == 1:
        chk_opt = np.array([chk_dz, np.nan])
        eps_opt = np.array([eps_state, np.nan])
    elif flag_conv == 2:
        chk_opt = np.array([np.nan, chk_cost])
        eps_opt = np.array([np.nan, eps_cost])

    # === Convergence check
    bool_conv = (
        (np.all(chk_feas_1 <= eps_feas_1) and np.all(chk_opt[0] <= eps_opt[0])) or
        (np.all(chk_feas_2 <= eps_feas_2) and np.all(chk_opt[1] <= eps_opt[1]))
    )

    # === Populate convergence summary
    conv_data.update({
        'bool_conv': bool_conv,
        'chk_dz': chk_opt[0],
        'chk_opt': np.nanmax(chk_opt),
        'chk_feas_term': chk_vb_term,
        'chk_feas_path': chk_vb_path,
        'chk_feas_nfz': chk_vb_nfz,
        'chk_feas_aux': chk_vb_aux,
        'chk_feas_dyn': chk_vb_dyn,
        'chk_feas': max(np.max(chk_feas_1), np.max(chk_feas_2)),
        'status': O['subprob'].status,
    })

    O['converged'] = bool_conv
    O['conv_data'] = conv_data
    return O


# Example usage
if __name__ == "__main__":
    # Define a dummy params dictionary for testing
    params = {
        'nz': 3,
        'n_ineq': 1,
        'n_path': 2,
        'n_nfz': 1,
        'n_aux': 1,
        'n_term': 1,
        'n_term_ineq': 1,
        'n_dyn': 2,
        'conv': {
            'setup': {
                'eps_state': 1e-6,
                'eps_cost': 1e-6,
                'eps_path': 1e-6,
                'eps_nfz': 1e-6,
                'eps_aux': 1e-6,
                'eps_term': 1e-6,
                'eps_defect': 1e-6,
                'eps_dyn': 1e-6,
                'ctcs_mult_state': 1.0,
                'ctcs_mult_cnst': 1.0
            }
        },
        'bools': {
            'ctcs': True
        },
        'nondim': {
            'M_state_d2nd': np.eye(3),
            'M_cost_d2nd': np.eye(1),
            'M_path_d2nd': np.eye(2),
            'M_nfz_d2nd': np.eye(1),
            'M_aux_d2nd': np.eye(1),
            'M_term_d2nd': np.eye(2),
            'M_cnst_d2nd': np.eye(3),
            'M_dyn_d2nd': np.eye(2)
        }
    }

    updated_params = set_convergence_tolerance(params)
    print(updated_params)


    # Define a dummy problem and O for testing
    problem = {
        'params': {
            'conv': {
                'eps_state': 1e-6,
                'Wconv_state': np.eye(3),
                'Wconv_state_vec': np.ones(3),
                'eps_cost': 1e-6,
                'eps_path': 1e-6,
                'Wconv_path': np.eye(3),
                'Wconv_path_vec': np.ones(3),
                'eps_nfz': 1e-6,
                'Wconv_nfz': np.eye(3),
                'Wconv_nfz_vec': np.ones(3),
                'eps_aux': 1e-6,
                'Wconv_aux': np.eye(3),
                'Wconv_aux_vec': np.ones(3),
                'eps_term': 1e-6,
                'Wconv_term': np.eye(3),
                'Wconv_term_vec': np.ones(3),
                'eps_defect': 1e-6,
                'Wconv_defect': np.eye(3),
                'Wconv_defect_vec': np.ones(3),
                'eps_dyn': 1e-6,
                'Wconv_dyn': np.eye(3),
                'Wconv_dyn_vec': np.ones(3)
            },
            'bools': {
                'ctcs': False,
                'flag_conv': 0
            },
            'path_idx': [0, 1, 2],
            'nfz_idx': [0, 1, 2],
            'aux_idx': [0, 1, 2]
        }
    }

    O = {
        'conv_data': {
            'soln': {'problem': 0},
            'cost_ref': 1.0,
            'vb_path':  np.zeros((3, 3)),
            'vb_nfz':   np.zeros((3, 3)),
            'vb_aux':   np.zeros((3, 3)),
            'vb_term':  np.zeros((3, 3)),
            'vb_dyn':   np.zeros((3, 3)),
            'defect':   np.zeros((3, 3))
        },
        'dz_s': np.zeros((3, 3)),
        'cost': 1.0,
        'cnst_path': np.zeros((3, 3))
    }

    result = check_convergence_tolerance(O, problem)
    print(result)