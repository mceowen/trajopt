import numpy as np

import numpy as np
print(">>> LOADED reentry_3dof_ghame.py <<<")
# this is vtol_wang. change it so it matches ghame on matlab

def config_params():
    """
    Configures parameters dictionary for reentry_3dof case (case_flag = 1)
    """

    params = {}


    # === Case setup ===
    params['nondim_on'] = False      
    params['case_flag'] = 1            # C:\Users\chris\hypersonic_entry_opt\config\config_main.m
    params['N']         = 40            # C:\Users\chris\hypersonic_entry_opt\entry_problems\missions\ghame\config_params_main.m
    params['n']         = 6             # 
    params['m']         = 2             # should this be 2  (was 1)
    params['T_init']    = 1700.0        # C:\Users\chris\hypersonic_entry_opt\entry_problems\missions\ghame\config_params_main.m

    # === Physical constants ===
    params['ge']        = 9.81          # C:\Users\chris\hypersonic_entry_opt\entry_problems\missions\ghame\config_params_planet.m
    params['re']        = 6378.137e3    
    params['mue']       = 3.986004418e14
    params['rhoe']      = 1.3
    params['H']         = 7e3
    params['beta']      = 1.0 / params['H']
    params['flags']     = {'earth_rot': 1}
    sidereal_day_s      = 23 * 3600 + 56 * 60 + 4
    params['Omega']     = 2 * np.pi / sidereal_day_s * params['flags']['earth_rot']

    # === Vehicle mass & reference geometry ===
    params['mass']      = 54431.0  # C:\Users\chris\hypersonic_entry_opt\entry_problems\missions\ghame\config_params_vehicle.m    
    params['Sref']      = 557.42  # [m^2]
    #params['ce']        = 0.5       # only used for case_flag = 3
    # area
    params['sigma_max']     = np.deg2rad(45)    # [rad], bank angle upper bound
    params['sigma_min']     = -np.deg2rad(45)   # [rad], bank angle lower bound
    params['sigma_dot_max'] = np.deg2rad(10)    # deg per sec
    params['alpha_hardmax'] = np.deg2rad(25)    # [rad], angle-of-attack hard upper bound
    params['alpha_hardmin'] = np.deg2rad(-5)    # [rad], angle-of-attack hard lower bound
    params['alpha_slack']   = np.deg2rad(0)     # [rad], angle-of-attack slack from given f(v)
    params['alpha_dot_max'] = np.deg2rad(1)     # deg per sec
    params['flags']         = {'aoa_vb': 0}     

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
    h0                  = 80e3              # [m], altitude IC
    theta0              = np.deg2rad(0)     # [rad], longitude IC
    phi0                = np.deg2rad(0)     # [rad], latitude IC
    v0                  = 5612              # [m/s], velocity IC
    gamma0              = np.deg2rad(-0.5)  # [rad], flight path angle IC
    psi0                = np.deg2rad(0)     # [rad], heading IC

    # create initial boundary condition vector
    params['z0'] = np.array([
        (params['re'] + h0) ,
        theta0 ,
        phi0 ,
        v0 ,
        gamma0 ,
        psi0 
    ])

    params['z0s'] = np.array([
        (params['re'] + h0) / nd,
        theta0 ,
        phi0 ,
        v0 / nv,
        gamma0,
        psi0
    ])


    # === Initial time grid ===
    # C:\Users\chris\hypersonic_entry_opt\entry_problems\missions\ghame\custom\generate_initial_guess.m
    dt_init             = (params['T_init'] / (params['N'] - 1)) * np.ones(params['N'] - 1)
    dt_init            = dt_init / params['nondim']['nt']  # nondimensionalized
    Ts_init             = params['T_init'] / params['nondim']['nt']
    t_init             = np.cumsum(np.insert(dt_init, 0, 0.0))  # size N

    params['dt_init']   = dt_init       # dimensional [s]
    params['dt_init']  = dt_init      # nondimensional
    params['Ts_init']   = Ts_init       # nondimensional
    params['t_init']   = t_init       # nondimensional

    # === Initial trajectory guess ===
    # params['z_init'] = np.tile(params['z0'], (params['N'], 1))       # (N, 6)

    return params



def extract_N(ts):
    N = 1 if isinstance(t, float) else (ts.shape[0] if ts.ndim == 1 else ts.shape[1])
    return N

def mass_thrust(t, z, nu, params):
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
        Tf      = nu[2] if N == 1 else nu[:, 2]
        mass    = z[6] if N == 1 else z[:, 6]
    else:
        Tf      = 0. / params['nondim']['nf']
        mass    = params['mass'] / params['nondim']['nm']

    return {
            'mass' : mass, 
            'Tf' : Tf
        }


def nonlinear_aero(t, z, nu, params, case_flag=None):

    # Extract params if "problem" parent struct is passed in
    if 'params' in params:
        params = params['params']

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

    for k in range(N):

        tk = ts if N == 1 else t[k]
        zk = z if N == 1 else z[k]
        uk = nu if N == 1 else nu[k]

        r, theta, phi, v, gamma, psi = z if N == 1 else z[k]
        # r, theta, phi, v, gamma, psi = zk[:6] old bad line

        # Extract thrust and mass
        force   = mass_thrust(tk, zk, uk, params)
        mass    = force['mass']
        Tf      = force['Tf']

        # mass_and_thrust = mass_thrust(tk, zk, uk, params) 
        # mass = mass_and_thrust['mass']

        # Extract control
        if case_flag == 1:
            alpha[k] = np.deg2rad(15);
            alpha_deg = np.rad2deg(alpha[k]);

        elif case_flag in (2, 3):
            alpha[k] = nu if N == 1 else nu[k, 1]
            alpha_deg = np.rad2deg(alpha[k])

        # COEFFICIENTS

        M   = v * nv / np.sqrt(1.4 * 287 * 239)       # TO-DO - REPLACE WITH FUNCTION CALL
        cl0 = 0.0052*np.log(M)-0.0334
        cl1 = 0.03* (M**(-0.49));
        cd0 = 0.0577*np.exp(-0.042*M)
        cd1 = 0.00879*np.log(M)-0.0192
        cd2 = 0.4521*(M**(0.4856))

        # AoA-DEPENDENT AERO COEFFICIENTS
        Cl[k] = cl0 + cl1*alpha_deg
        Cd[k] = cd0 + (cd1 * Cl[k]) + (cd2 * (Cl[k]**2))

        # Compute lift and drag
        rho     = rhoe * np.exp(-beta * (params['nondim']['nd'] * r - re))
        L[k]    = (B / mass) * rho * Cl[k] * v**2
        D[k]    = (B / mass) * rho * Cd[k] * v**2

        return {
            'L': L,
            'D': D,
            'Cl': Cl,
            'Cd': Cd,
            'alpha': alpha,
            'rho': rho
        }

"""
    if N == 1:
        return {
            'L': float(L[0]),
            'D': float(D[0]),
            'Cl': float(Cl[0]),
            'Cd': float(Cd[0]),
            'alpha': float(alpha[0])
        }
    
    else: 
        return {
            'L': L,
            'D': D,
            'Cl': Cl,
            'Cd': Cd,
            'alpha': alpha
        }"""
    

# this is the vtol_wang nonlinear aero
"""
def nonlinear_aero(t, z, nu, params, case_flag=None):
    
    print("DEBUG nonlinear_aero CALLED")

    
    Compute nonlinear aerodynamic coefficients and lift/drag forces.

    Parameters:
    ts (numpy.ndarray): Time vector.
    zs (numpy.ndarray): State vector.
    us (numpy.ndarray): Control input.
    params (dict): Dictionary containing parameters.
    case_flag (int, optional): Case flag for determining the type of coefficients. Default is None.

    Returns:
    dictionary: Aerodynamic coefficients and angle of attack.
    

    # Extract params if "problem" parent struct is passed in
    if 'params' in params:
        params = params['params']

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

    # Setup coefficient values
    kl1         = -0.041065
    kl2         = 0.016292
    kl3         = 0.0002602
    kd1         = 0.080505
    kd2         = -0.03026
    kd3         = 0.86495
    kalph       = 0.20705 / (340**2)
    vlim        = 4570
    alphlim_deg = 40

    if case_flag == 1:
        # Velocity-dependent polynomial coefficients
        Kd1     = kd1
        Kd2     = kd2
        Kd3     = kd3
        Kl1     = kl1 + kl2 * alphlim_deg + kl3 * alphlim_deg**2
        Kl2     = -kl2 * kalph - 2 * kl3 * alphlim_deg * kalph
        Kl3     = kl3 * kalph**2
    elif case_flag in [2, 3]:
        # AOA-dependent polynomial coefficients
        d2r     = 1  # /(pi/180)
        Kd1h    = kd1 + kd2 * kl1 + kd3 * kl1**2
        Kd2h    = (kd2 * kl2 + 2 * kd3 * kl1 * kl2) * d2r
        Kd3h    = (kd2 * kl3 + 2 * kd3 * kl1 * kl3 + kd3 * kl2**2) * d2r**2
        Kd4h    = (2 * kd3 * kl2 * kl3) * d2r**3
        Kd5h    = (kd3 * kl3**2) * d2r**4
        Kl1h    = kl1
        Kl2h    = kl2 * d2r**2
        Kl3h    = kl3 * d2r**3
    else:
        raise ValueError('Undefined case_flag!')

    for k in range(N):
        # Extract states and controls
        tk = ts if N == 1 else t[k]
        zk = z if N == 1 else z[k]
        uk = nu if N == 1 else nu[k]
        r, theta, phi, v, gamma, psi = z if N == 1 else z[k]

        # Extract thrust and mass
        force   = mass_thrust(tk, zk, uk, params)
        mass    = force['mass']
        Tf      = force['Tf']

        # Extract control
        if case_flag == 1:
            # Velocity-dependent coefficients
            v_sat       = min(v * nv, vlim)
            Cl[k]       = Kl1 + Kl2 * (v_sat - vlim)**2 + Kl3 * (v_sat - vlim)**4
            Cd[k]       = Kd1 + Kd2 * Cl[k] + Kd3 * Cl[k]**2
            alpha[k]    = np.deg2rad(alphlim_deg - kalph * (min(v * nv, vlim) - vlim)**2)
        elif case_flag in [2, 3]:
            alpha[k]    = nu if N == 1 else nu[k,1]
            alpha_deg   = np.rad2deg(alpha[k])
            Cl[k]       = Kl1h + Kl2h * alpha_deg + Kl3h * alpha_deg**2
            Cd[k]       = Kd1h + Kd2h * alpha_deg + Kd3h * alpha_deg**2 + Kd4h * alpha_deg**3 + Kd5h * alpha_deg**4

        # Compute lift and drag
        rho     = rhoe * np.exp(-beta * (params['nondim']['nd'] * r - re))
        L[k]    = (B / mass) * rho * Cl[k] * v**2
        D[k]    = (B / mass) * rho * Cd[k] * v**2

    return {
        'L': L,
        'D': D,
        'Cl': Cl,
        'Cd': Cd,
        'alpha': alpha,
        'rho': rho
    }
"""

def system_dynamics(t, z, nu, params, t_vec=None):
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
    r, theta, phi, v, gamma, psi = z
    
    # Extract controls 
    """if t_vec is None:
        us2 = nu
    else:
        us2 = np.zeros(m)
        for i in range(m-1):
            us2[i] = np.interp(t, t_vec, nu[:, i])"""
    
    #us2 = nu

    """
    # Extract bank angle
    sigma   = nu2 if isinstance(us2, float) else us2[0]"""

    #Extract controls 
    """if np.ndim(nu) == 0:                # scalar (unlikely here)
        us2 = np.array([us])
    elif np.ndim(nu) == 1:              # single control vector (σ, α)
        us2 = nu
    elif np.ndim(nu) == 2:              # full trajectory (m, N)
        # pick the closest column for current t
        k = np.argmin(np.abs(t_vec - ts)) if t_vec is not None else 0
        us2 = nu[:, k]
    else:
        raise ValueError("Unexpected control input shape for us")"""

    if np.ndim(nu) == 1:
        us2 = nu
    elif np.ndim(nu) == 2:
        if us.shape[1] == 1:
            us2 = nu[:, 0]
        else:
            k = np.argmin(np.abs(t_vec - ts)) if t_vec is not None else 0
            us2 = nu[:, k]
    else:
        raise ValueError(f"Unexpected control shape: {us.shape}")


    # Now safely extract individual controls
    sigma = float(us2[0])   # bank angle
    alpha = float(us2[1])   # angle of attack

    # Determine lift and drag coefficients from velocity
    aero    = nonlinear_aero(t, z, us2, params)
    L       = aero['L']
    D       = aero['D']
    Cl      = aero['Cl']
    Cd      = aero['Cd']
    alpha   = aero['alpha']

    # Extract mass and thrust
    force   = mass_thrust(t, z, us2, params)
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