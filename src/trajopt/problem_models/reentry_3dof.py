import numpy as np


def config_params():
    """
    Configures parameters dictionary for reentry_3dof case (case_flag = 1)
    """

    params = {}

    # === System setup ===
    params['case_flag'] = 1
    params['n'] = 6
    params['m'] = 1
    params['T_init'] = 1700.0  # [s], total propagation time

    # === Physical constants ===
    params['ge'] = 9.81  # [m/s^2]
    params['re'] = 6378.137e3  # [m]
    params['rhoe'] = 1.3       # [kg/m^3]
    params['H'] = 7e3          # [m], scale height
    params['beta'] = 1.0 / params['H']
    params['mue'] = 3.986004418e14  # [m^3/s^2], gravitational constant

    # === Earth rotation (set to 1 for enabled) ===
    params['bools'] = {}
    params['bools']['earth_rot'] = 1
    day_seconds = 23 * 3600 + 56 * 60 + 4  # sidereal day in seconds
    params['Omega'] = 2 * np.pi / day_seconds * params['bools']['earth_rot']

    # === Initial state ===
    h0 = 100e3  # [m]
    theta0 = np.deg2rad(0)
    phi0 = np.deg2rad(0)
    v0 = 7450  # [m/s]
    gamma0 = np.deg2rad(-0.5)
    psi0 = np.deg2rad(0)

    params['z0'] = np.array([params['re'] + h0, theta0, phi0, v0, gamma0, psi0])  # shape (6,)

    # === Vehicle properties ===
    mass = 1000.0  # [kg]
    params['mass'] = mass
    params['ce'] = 0.5  # optional, only relevant for case_flag = 3

    # === Nondimensionalization ===
    nt = np.sqrt(params['re'] / params['ge'])
    nv = np.sqrt(params['re'] * params['ge'])
    nf = mass * params['ge']
    nm = mass
    nd = params['re']
    na = params['ge']
    nm_dot = mass / nt

    params['nondim'] = {
        'nt': nt,
        'nt_inv': 1.0 / nt,
        'nd': nd,
        'nv': nv,
        'na': na,
        'nm': nm,
        'nm_dot': nm_dot,
        'nf': nf,
    }

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
    tuple: Mass and thrust.
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

    return mass, Tf

def nonlinear_aero(ts, zs, us, params, case_flag=None):
    """
    Compute nonlinear aerodynamic coefficients and lift/drag forces.

    Parameters:
    ts (numpy.ndarray): Time vector.
    zs (numpy.ndarray): State vector.
    us (numpy.ndarray): Control input.
    params (dict): Dictionary containing parameters.
    case_flag (int, optional): Case flag for determining the type of coefficients. Default is None.

    Returns:
    tuple: Aerodynamic coefficients and angle of attack.
    """
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
        tk = ts if N == 1 else ts[k]
        zk = zs if N == 1 else zs[k]
        uk = us if N == 1 else us[k]
        r, theta, phi, v, gamma, psi = zs if N == 1 else zs[k]

        # Extract thrust and mass
        mass, _ = mass_thrust(tk, zk, uk, params)

        # Extract control
        if case_flag == 1:
            # Velocity-dependent coefficients
            v_sat = min(v * nv, vlim)
            Cl[k] = Kl1 + Kl2 * (v_sat - vlim)**2 + Kl3 * (v_sat - vlim)**4
            Cd[k] = Kd1 + Kd2 * Cl[k] + Kd3 * Cl[k]**2
            alpha[k] = np.deg2rad(alphlim_deg - kalph * (min(v * nv, vlim) - vlim)**2)
        elif case_flag in [2, 3]:
            alpha[k] = us[1] if N == 1 else us[k,1]
            alpha_deg = np.rad2deg(alpha[k])
            Cl[k] = Kl1h + Kl2h * alpha_deg + Kl3h * alpha_deg**2
            Cd[k] = Kd1h + Kd2h * alpha_deg + Kd3h * alpha_deg**2 + Kd4h * alpha_deg**3 + Kd5h * alpha_deg**4

        # Compute lift and drag
        rho = rhoe * np.exp(-beta * (params['nondim']['nd'] * r - re))
        L[k] = (B / mass) * rho * Cl[k] * v**2
        D[k] = (B / mass) * rho * Cd[k] * v**2

    return L, D, Cl, Cd, alpha

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
    
    # Extract controls 
    if t_vec is None:
        us2 = us
    else:
        us2 = np.zeros(m)
        for i in range(m-1):
            us2[i] = np.interp(ts, t_vec, us[:, i])

    # Extract bank angle
    sigma = us2 if isinstance(us2, float) else us2[0]

    # Determine lift and drag coefficients from velocity
    L, D, Cl, Cd, alpha = nonlinear_aero(ts, zs, us2, params)

    # Extract mass and thrust
    mass, Tf = mass_thrust(ts, zs, us2, params)

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
    xDot[4]     = (1 / v) * ((Tf / mass) * sa + L) * cs + (v**2 - Kg / r) * cg / r + 2 * Om * cp * sps + Om**2 * r * (1 / v) * cp * (cg * cp + sg * cps * sp)
    # psi_dot
    xDot[5]     = (1 / v) * ((Tf / mass) * sa + L) * ss / cg + v**2 * cg * sps * tp / r - 2 * Om * (tg * cps * cp - sp) + Om**2 * r * (1 / (v * cg)) * sps * sp * cp

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
    # Define dummy data for testing
    # note can turn to column vector via: 
    #  ts      = np.array([0.0])[:, np.newaxis]
    ts          = np.array([0.0])
    zs          = np.array([1, 0.5, 0.2, 3000, 0.05, 0.1, 1000])
    us          = np.zeros((100, 3)) 
    us[:, 0]    = np.linspace(0.1, 0.2, 100)
    us[:, 1]    = np.linspace(0.1, 0.2, 100)
    us[:, 2]    = np.linspace(0.1, 0.2, 100)

    t_vec   = np.linspace(0, 10, 100)

    params  = {
        're': 6371e3,
        'rhoe': 1.225,
        'beta': 0.1,
        'B': 0.5,
        'Omega_s': 7.2921159e-5,
        'kg': 3.986004418e14,
        'n': 6,
        'm': 1,
        'case_flag': 1,
        'mass': 1000,
        'ce': 0.5,
        'nondim': {'nv': 1.0, 'nd': 1.0, 'nf': 1.0, 'nm': 1.0},
    }

    
    xDot    = system_dynamics(ts, zs, us, params, t_vec)
    print(xDot)