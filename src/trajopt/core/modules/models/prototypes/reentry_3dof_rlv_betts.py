import numpy as np
from scipy.integrate import solve_ivp

print(">>> LOADED reentry_3dof_ghame.py <<<")
# this is vtol_wang. change it so it matches ghame on matlab

"""
   0.000 s          1 calls   C:\Users\chris\hypersonic_entry_opt\entry_problems\missions\rlv_betts\config_params_main.m>config_params_main
   0.000 s          1 calls   C:\Users\chris\hypersonic_entry_opt\entry_problems\missions\rlv_betts\config_params_planet.m>config_params_planet
   0.005 s          1 calls   C:\Users\chris\hypersonic_entry_opt\entry_problems\missions\rlv_betts\config_params_vehicle.m>config_params_vehicle
   1.529 s          1 calls   C:\Users\chris\hypersonic_entry_opt\entry_problems\missions\rlv_betts\custom\generate_initial_guess.m>generate_initial_guess
   0.014 s          1 calls   C:\Users\chris\hypersonic_entry_opt\entry_problems\missions\rlv_betts\config_params_convergence.m>config_params_convergence
   0.000 s          1 calls   C:\Users\chris\hypersonic_entry_opt\entry_problems\missions\rlv_betts\config_params_time.m>config_params_time
 203.538 s    1774915 calls   C:\Users\chris\hypersonic_entry_opt\entry_problems\missions\rlv_betts\aero\nonlinear_aero.m>nonlinear_aero
 382.006 s     593023 calls   C:\Users\chris\hypersonic_entry_opt\entry_problems\missions\rlv_betts\aero\analytical_aero.m>analytical_aero
 139.509 s         25 calls   C:\Users\chris\hypersonic_entry_opt\entry_problems\missions\rlv_betts\custom\analytical_cost.m>analytical_cost
   1.134 s       7429 calls   C:\Users\chris\hypersonic_entry_opt\entry_problems\missions\rlv_betts\custom\generate_initial_guess.m>@(t,x)system_dynamics(t,x,us_init,params,ts_init)
   0.003 s         76 calls   C:\Users\chris\hypersonic_entry_opt\entry_problems\missions\rlv_betts\config_params_vehicle.m>@(t,z,u)-z(3,end)
   0.000 s         25 calls   C:\Users\chris\hypersonic_entry_opt\entry_problems\missions\rlv_betts\custom\custom_inputs.m>custom_inputs
   0.000 s         25 calls   C:\Users\chris\hypersonic_entry_opt\entry_problems\missions\rlv_betts\custom\custom_constraints.m>custom_constraints
   0.000 s         25 calls   C:\Users\chris\hypersonic_entry_opt\entry_problems\missions\rlv_betts\custom\custom_outputs.m>custom_outputs


RLV Betts calls in matlab version
    config_params_main.m        - done
    config_params_planet.m      - done
    config_params_vehicle.m     - done
    generate_initial_guess.m    - done
    straight_line_initial_guess - done
    config_params_convergence.m - check matlab init_params_struct and scp
    config_params_time.m        -        
    nonlinear_aero.m            - done
    analytical_aero.m           - 
    analytical_cost.m           -  

    generate_initial_guess.m>@(t,x)system_dynamics(t,x,us_init,params,ts_init)
    config_params_vehicle.m>@(t,z,u)-z(3,end)
    custom_inputs.m>custom_inputs
    custom_constraints.m>custom_constraints
    custom_outputs.m>custom_outputs
"""

def config_params():
    """
    Configures parameters dictionary for reentry_3dof case (case_flag = 1)
    """

    params = {}


    # === Case setup ===
    params['nondim_on'] = False      
    params['case_flag'] = 1            
    params['N']         = 150            
    params['n']         = 6             # 
    params['m']         = 2             # should this be 2  (was 1)
    params['T_init']    = 2000.0        

    params['flags'] = {
        'earth_rot': 1,
        'init_ctrl': 0,
        'aoa_vb': 0,
        'buff_dyn': 'term',
        'ctcs': 0
    }

    # === Physical constants ===
    params['ge']        = 9.81          
    params['re']        = 6378.137e3    
    params['mue']       = 3.986004418e14
    params['rhoe']      = 1.225570827014494
    params['H']         = 7254.24
    params['beta']      = 1.0 / params['H']
    params['day']       = 23*3600+56*60+4
    params['omega']     = 2*np.pi/params['day'] * params['flags']['earth_rot']
    params['mu_earth']  = 3.986031954093051e14

    # === Define scaling choices === 
    # buff_dyn is in params['flags']

    # === Vehicle mass & reference geometry ===
    params['Kq']            = 1.2035e-5             # [-], htrt coeff
    params['mass']          = 92079.2525560557      # RLV mass    
    params['Sref']          = 249.9091776           # [m^2], RLV reference area
    #params['ce']           = 0.5                   # only used for case_flag = 3
    params['sigma_max']     = np.deg2rad(1)         # [rad], bank angle upper bound
    params['sigma_min']     = np.deg2rad(-90)       # [rad], bank angle lower bound
    params['sigma_dot_max'] = []                    # deg per sec
    params['alpha_hardmax'] = np.deg2rad(60)        # [rad], angle-of-attack hard upper bound
    params['alpha_hardmin'] = np.deg2rad(0)         # [rad], angle-of-attack hard lower bound
    params['alpha_slack']   = np.deg2rad(0)         # [rad], angle-of-attack slack from given f(v)
    params['alpha_dot_max'] = []                    # deg per sec
    params['z_max_idx']     = []
    params['z_min_idx']     = []

    h_max                   = []
    r_max                   = h_max + params['re']
    theta_max               = []
    phi_max                 = []
    v_max                   = []
    gamma_max               = []
    psi_max                 = []

    h_min                   = []
    r_min                   = h_min + params['re']
    theta_min               = []
    phi_min                 = []
    v_min                   = []
    gamma_min               = []
    psi_min                 = []



    # === Nondimensionalization ===
    if params['nondim_on']:
        # Compute nondimensional values using local variables
        nt      = np.sqrt(params['re'] / params['ge'])
        nt_inv  = 1.0 / nt
        nd      = params['re']
        nv      = np.sqrt(params['re'] * params['ge'])
        na      = params['ge']
        nm      = params['mass']
        nm_dot  = nm / nt
        nf      = nm * na
    else: 
        # Identity scalings for fully dimensional case
        nt = nt_inv = nd = nv = na = nm = nm_dot = nf = 1.0

    # Assign to params
    params['nondim'] = {
        'nt': nt,
        'nt_inv': nt_inv,
        'nd': nd,
        'nv': nv,
        'na': na,
        'nm': nm,
        'nm_dot': nm_dot,
        'nf': nf,
    }

    # === Scaled constants ===
    params['kg']        = params['mue'] / (na * nd**2)
    params['B']         = (nd / nm) * (params['Sref'] / 2)
    params['Omega_s']   = params['Omega'] * nt  # scaled Earth rotation

    # === Initial state
    h0                  = 80e3                  # [m], altitude IC
    r0                  = (h0 + params['re'])   # % [m], radial position
    theta0              = np.deg2rad(0)         # [rad], longitude IC
    phi0                = np.deg2rad(0)         # [rad], latitude IC
    v0                  = 7802.88               # [m/s], velocity IC
    gamma0              = np.deg2rad(-1)        # [rad], flight path angle IC
    psi0                = np.deg2rad(90)        # [rad], heading IC

    # initial boundary condition index
    params['z0_idx'] = np.arange(params['n'])

    # create initial boundary condition vector
    params['z0'] = np.array([
        (params['re'] + h0) ,
        theta0 ,
        phi0 ,
        v0 ,
        gamma0 ,
        psi0 
    ])

    # non-dimensionalized initial state vector
    params['z0s'] = np.array([
        (params['re'] + h0) / nd,
        theta0 ,
        phi0 ,
        v0 / nv,
        gamma0,
        psi0
    ])

    # === Final state
    hf                  = 24.384e3              # [m], altitude final condition
    rf                  = (hf + params['re'])   # [m], radial position final condition
    thetaf              = []                    # [rad], lon final condition (~8000km)
    phif                = []                    # [rad], lat final condition (~500km)
    vf                  = 762                   # [m/s], velocity final condition
    gammaf              = np.deg2rad(-5)        # [rad], fpa final condition
    psif                = []                    # [rad], hdg final condition 

    # final boundary condition index
    params['zf_idx'] = np.arange(params['n'])

    # create terminal boundary condition vector
    params['zf'] = np.array([
        (params['re'] + hf),
        thetaf,
        phif,
        vf,
        gammaf,
        psif
    ])

    # non-dimensionalized final state vector
    params['z0s'] = np.array([
        (params['re'] + h0) / nd,
        thetaf ,
        phif ,
        vf / nv,
        gammaf,
        psif
    ])

    params['cost'] = lambda t, z, u: -z[2, -1]
    params['ncost'] = np.rad2deg(1)
    params['cost_name'] = 'Terminal latitude [deg]'


    # === Initial time grid ===
    dt_init             = (params['T_init'] / (params['N'] - 1)) * np.ones(params['N'] - 1)
    dts_init            = dt_init / params['nondim']['nt']  # nondimensionalized
    Ts_init             = params['T_init'] / params['nondim']['nt']
    ts_init             = np.cumsum(np.insert(dts_init, 0, 0.0))  # size N

    # === Initial control === 
    sigmas_init = np.linspace(np.deg2rad(-30), np.deg2rad(-30), params['N'])
    alphas_init = np.linspace(np.deg2rad(10), np.deg2rad(10), params['N'])

    if params['case_flag'] == 1:
        us_init = sigmas_init       
        params['alpha_nom'] = alphas_init
    
    elif params['case_flag'] in [2, 3]:
        us_init = np.vstack(sigmas_init, alphas_init)
    else:
        raise ValueError('Undefined case_flag!')

    params['dt_init']   = dt_init       # dimensional [s]
    params['dts_init']  = dts_init      # nondimensional
    params['Ts_init']   = Ts_init       # nondimensional
    params['ts_init']   = ts_init       # nondimensional

    # === Initial trajectory guess ===
    # params['zs_init'] = np.tile(params['z0'], (params['N'], 1))       # (N, 6)

    # init_params_struct from matlab
    params['zi'] = params['z0s']

    # for generate initial guess


    return params


def extract_N(ts):
    N = 1 if isinstance(ts, float) else (ts.shape[0] if ts.ndim == 1 else ts.shape[1])
    return N

def mass_thrust(ts, zs, us, params):
    """
    Compute mass and thrust.

    Parameters:
    ts (numpy.ndarray): Time vector.
    zs (numpy.ndarray): State vector.
    us (numpy.ndarray): Control input.
    problem (dict): Dictionary containing parameters.

    Returns:
    dictionary: Mass and thrust.
    """
    # Extract params if "problem" parent struct is passed in
    params  = params['params'] if 'params' in params else params
    N       = extract_N(ts)

    if params['case_flag'] == 3:    
        Tf      = us[2] if N == 1 else us[:, 2]
        mass    = zs[6] if N == 1 else zs[:, 6]
    else:
        Tf      = 0. / params['nondim']['nf']
        mass    = params['mass'] / params['nondim']['nm']

    return {
            'mass' : mass, 
            'Tf' : Tf
        }


def nonlinear_aero(ts, zs, us, params, case_flag=None):
    
    # Extract params if "problem" parent struct is passed in
    if case_flag is None:
        case_flag = params['case_flag']

    # Extract key params
    nv      = params['nondim']['nv']
    B       = params['B']
    rhoe    = params['rhoe']
    beta    = params['beta']
    re      = params['re']
    N       = extract_N(ts)

    # Initialize output vectors
    Cl      = np.zeros(N)
    Cd      = np.zeros(N)
    alpha   = np.zeros(N)
    L       = np.zeros(N)
    D       = np.zeros(N)

    # generate common aero constants
    # setup coefficient values
    k_cl = np.array([-0.2070, 1.6756])
    k_cd = np.array([0.0785, -0.3529, 2.0400])

    for k in range(N):
        
        # Extract states and controls
        tk = ts if N == 1 else ts[k]
        zk = zs if N == 1 else zs[k]
        uk = us if N == 1 else us[k]
        r, theta, phi, v, gamma, psi = zs if N == 1 else zs[k]

        # Extract thrust and mass
        force   = mass_thrust(tk, zk, uk, params)
        mass    = force['mass']
        Tf      = force['Tf']

        if case_flag == 1:
            
            #VELOCITY-DEPENDENT COEFFICIENTS
            # Determine lift and drag coefficients from velocity, and their derivatives

            if N > params['N']:
                alpha[k] = np.interp(params['ts_init'], params['alpha_nom'])
            else:
                alpha[k] = params['alpha_nom'][k]
            
            Cl[k] = k_cl[0] + k_cl[1]*alpha[k]
            Cd[k] = k_cd[0] + k_cd[1]*alpha[k] + k_cd[2]*alpha[k]**2


        elif case_flag in [2, 3]:

            alpha[k] = us[1,k]
            Cl[k] = k_cl[0] + k_cl[1]*alpha[k]
            Cd[k] = k_cd[0] + k_cd[1]*alpha[k] + k_cd[2]*alpha[k]**2

        rho = rhoe * np.exp(-beta * (params['nondim']['nd'] * r - re))
        L[k] = (B/mass) * rho * Cl[k] * v**2
        D[k] = (B/mass) * rho * Cd[k] * v**2

    return {
        'L': L,
        'D': D,
        'Cl': Cl,
        'Cd': Cd,
        'alpha': alpha,
        'rho': rho
    }


def system_dynamics(ts, zs, us, params, t_vec=None):
    """
    Nonlinear polar 3DoF hypersonic entry rotating earth dynamics

    Parameters:
    ts (float): Current time.
    zs (numpy.ndarray): State vector.
    us (numpy.ndarray): Control input.
    params (dict): Dictionary containing parameters.
    t_vec (numpy.ndarray, optional): Time vector for interpolation. Default is None.

    Returns:
    numpy.ndarray: Derivative of the state vector.
    """

    if 'params' in params:
        params = params['params']

    # Extract constant param values from struct
    re          = params['re']
    rhoe        = params['rhoe']
    Om          = params['Omega_s']
    Kg          = params['kg']
    n           = int(params['n'])
    m           = int(params['m'])
    case_flag   = params['case_flag']
    N           = extract_N(ts)
    # Extract states
    r, theta, phi, v, gamma, psi = zs

    print(f"case_flag = {case_flag}, params keys = {params.keys()}")
    
    # Extract controls 
    us2 = us

    # Now safely extract individual controls
    sigma = float(us2[0])   # bank angle
    alpha = float(us2[1])   # angle of attack

    # Determine lift and drag coefficients from velocity
    aero    = nonlinear_aero(ts, zs, us2, params)
    L       = aero['L']
    D       = aero['D']
    Cl      = aero['Cl']
    Cd      = aero['Cd']
    alpha   = aero['alpha']

    # Extract mass and thrust
    force   = mass_thrust(ts, zs, us2, params)
    mass    = force['mass']
    Tf      = force['Tf']

    # Extract sines and cosines of various values
    cp          = np.cos(phi)
    sp          = np.sin(phi)
    tp          = np.tan(phi)
    cg          = np.cos(gamma)
    sg          = np.sin(gamma)
    tg          = np.tan(gamma)
    cps         = np.cos(psi)
    sps         = np.sin(psi)

    cs          = np.cos(sigma)
    ss          = np.sin(sigma)
    ca          = np.cos(alpha)
    sa          = np.sin(alpha)
    
    # Update states
    xDot        = np.zeros(n)

    # r_dot
    xDot[0]     = v * sg
    # theta_dot
    xDot[1]     = v * cg * sps / (r * cp)
    # phi_dot
    xDot[2]     = v * cg * cps / r
    # v_dot
    xDot[3]     = (Tf / mass) * ca - D - Kg * sg / r**2 + Om**2 * r * cp * (sg * cp - cg * sp * cps)
    # gamma_dot
    print("DEBUG types:", type(alpha), type(L), type(D), type(sigma), type(mass))
    xDot[4]     = (1 / v) * ( ((Tf / mass) * sa + L) * cs + (v**2 - Kg / r) * cg / r ) + 2 * Om * cp * sps + Om**2 * r * (1 / v) * cp * (cg * cp + sg * cps * sp)
    # psi_dot
    xDot[5]     = (1 / v) * ( ((Tf / mass) * sa + L) * ss / cg + v**2 * cg * sps * tp / r ) - 2 * Om * (tg * cps * cp - sp) + Om**2 * r * (1 / (v * cg)) * sps * sp * cp

    if case_flag == 3:
        ce = params['ce']
        # m_dot
        xDot[6] = ce * Tf

    if isinstance(r, (int, float)):
        if r <= 1:
            xDot = np.zeros(n)
    elif np.any(np.isnan(xDot)) or np.any(np.isinf(xDot)):
        raise ValueError("NaN or Inf values encountered in xDot")

    return xDot



# Example usage
if __name__ == "__main__":
    print("This module provides functions for reentry 3DoF dynamics and aerodynamics.")



def straight_line_initial_guess(params):
    params['dt_init'] = (params['T_init'] / (params['N'] - 1)) * np.ones(1, params['N'] - 1)
    params['Ts_init'] = params['T_init'] / params['nondim']['nt']
    params['dts_init'] = params['dt_init'] / params['nondim']['nt']
    ts_init = np.cumsum([0, params['dts_init']])

    # initial state
    if 'zf_init' in params:
        zs_init = np.zeros((params['n'], params['N']))
        for i_state in params['n']:
            zs_init[i_state, :] = np.linspace(params['z0'][i_state], 
                                              params['zf_init'][i_state],
                                              params['N']
            )
    else:
        zs_init = np.zeros((params['n'], params['N']))
        for i_state in params['n']:
            if i_state in params['zi_idx'] & i_state in params['zf_idx']:
                zs_init[i_state,:] = np.linspace(params['z0'](i_state), 
                                                 params['zf'](i_state),
                                                 params['N']
                )

    for i_state in params['n']:
        us_init = np.zeros((params['n'], params['N']))
        us_init[i_state, :] = np.zeros(params['N'])

    params['ts_init'] = ts_init
    params['zs_init'] = zs_init
    params['us_init'] = us_init

    
        
        





# Generate initial guess
def generate_initial_guess(params):

    bool_init_opt = 0

    # initialization trajectory
    params['dt_init'] = (params['T_init']/(params['N'] - 1)) * np.ones(1, params['N'] - 1)
    params['Ts_init'] = params['T_init']/params['nondim']['nt']
    params['dts_init'] = params['dt_init']/params['nondim']['nt']
    ts_init = np.cumsum(np.concatenate(([0], params['dts_init'])))

    # initial control
    sigmas_init = np.linspace(np.deg2rad(-30), np.deg2rad(-30), params['N'])
    alphas_init = np.linspace(np.deg2rad(10), np.deg2rad(10), params['N'])

    if params['case_flag'] == 1:
        us_init = alphas_init
        params['alpha_nom'] = alphas_init

    elif params['case_flag'] in [2, 3]:
        us_init = np.vstack(sigmas_init, alphas_init)

    else:
        raise ValueError('Undefined case_flag!')
    
    if params['flags']['buff_dyn'] in ['term']:
        sol = solve_ivp(lambda t, x: system_dynamics(t, x, us_init, params, ts_init), 
                        (ts_init[0], ts_init[-1]),
                        params['z0s'],
                        t_eval=ts_init, 
                        method='RK45',
                        rtol=1e-12,
                        atol=1e-12
        )

    else:
        params['zf_init']                   = np.zeros(params['n'])
        params['zf_init'][params['zf_idx']] = params['zf']
        params['zf_init'][1:3, None]        = np.deg2rad([10, 10])[:, None]
        params                              = straight_line_initial_guess(params)

        zs_init = params['zs_init']

    if ts_init.shape[1] == 2:
        zs_init = [zs_init[:, 0], zs_init[:, -1]]

    aero_init = analytical_aero(ts_init, zs_init, us_init, params)

    params['ts_init'] = ts_init
    params['zs_init'] = zs_init
    params['us_init'] = us_init

    # create initial aero
    params['aero_init'] = aero_init

    """
    if params['flags']['ctcs']:
        params = ctcs_initial_guess(params)
    """

def analytical_aero(params):
    pass

