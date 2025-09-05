import numpy as np

import numpy as np

def config_params():
    """
    Configures parameters dictionary for reentry_3dof case (case_flag = 1)
    """

    params = {}


    # === Case setup ===
    params['nondim_on'] = False
    params['case_flag'] = 1
    params['N']         = 40
    params['n']         = 6
    params['m']         = 1
    params['T_init']    = 500.0

    # === Physical constants ===
    params['ge']    = 3.73                          # [m/s^2]
    params['re']    = 3380e3                        # Mars radius [m]
    params['rhoe']  = 0.020                         # Surface atmospheric density [kg/m^3]
    params['H']     = 11.1e3                        # Scale height [m]
    params['beta']  = 1.0 / params['H']             # Inverse scale height
    params['mue']   = params['ge'] * params['re']**2  # Gravitational parameter for Mars
    # No rotation for now (can adjust later)
    params['bools'] = {'earth_rot': 0}
    params['Omega'] = 0.0

    # === Vehicle mass & reference geometry ===
    params['mass']  = 2900           # kg (MSL landed mass ~900, entry mass ~2900)
    params['Sref']  = 15.9           # m^2 (MSL aeroshell reference area)
    params['LD']    = 0.24           # Ballistic coefficient L/D
    params['bc']    = 120            # Ballistic coefficient β = m / (Cd * S)

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
    h0      = 120e3                  # Entry altitude [m]
    theta0  = np.deg2rad(0)
    phi0    = np.deg2rad(0)
    v0      = 5500                   # Entry velocity [m/s]
    gamma0  = np.deg2rad(-14.5)      # Entry flight path angle [rad]
    psi0    = 0.0                    # Mars doesn't use heading in this 2D sim

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
    dt_init             = (params['T_init'] / (params['N'] - 1)) * np.ones(params['N'] - 1)
    dts_init            = dt_init / params['nondim']['nt']  # nondimensionalized
    Ts_init             = params['T_init'] / params['nondim']['nt']
    ts_init             = np.cumsum(np.insert(dts_init, 0, 0.0))  # size N

    params['dt_init']   = dt_init       # dimensional [s]
    params['dts_init']  = dts_init      # nondimensional
    params['Ts_init']   = Ts_init       # nondimensional
    params['ts_init']   = ts_init       # nondimensional

    # === Initial trajectory guess ===
    # params['zs_init'] = np.tile(params['z0'], (params['N'], 1))       # (N, 6)

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
    
    Tf      = 0. / params['nondim']['nf']
    mass    = params['mass'] / params['nondim']['nm']

    return {
            'mass' : mass, 
            'Tf' : Tf
        }

def reference_bank_angle(t, z, params):
    """
    Reference bank angle schedule for Mars, based on velocity.

    Parameters:
        t : float or np.ndarray
            Time (not used here, kept for compatibility)
        z : np.ndarray
            State(s), shape (6,) for single state or (N, 6) for multiple
        params : dict
            Parameter dictionary (not used currently)

    Returns:
        sigma : float or np.ndarray
            Bank angle(s) in radians
    """
    # Handle single state vector (shape (6,))
    if z.ndim == 1:
        v = z[3]
        if v >= 3500:
            return np.deg2rad(75)
        elif v <= 1500:
            return np.deg2rad(50)
        else:
            return np.deg2rad(50 + (75 - 50) * (v - 1500) / (3500 - 1500))

    # Handle batched input (shape (N, 6))
    elif z.ndim == 2:
        v = z[:, 3]
        sigma = np.where(
            v >= 3500,
            np.deg2rad(75),
            np.where(
                v <= 1500,
                np.deg2rad(50),
                np.deg2rad(50 + (75 - 50) * (v - 1500) / (3500 - 1500))
            )
        )
        return sigma

    else:
        raise ValueError(f"Unexpected shape for z: {z.shape}")

def nonlinear_aero(ts, zs, us, params, case_flag=None):
    """
    Compute aerodynamic parameters (Lift, Drag, Cl, Cd, alpha, density)
    for high-speed Mars entry dynamics, vectorized over time steps.

    Parameters:
        ts (array or float): Time(s), ignored internally except for shape
        zs (array): State vector(s) [r, theta, phi, v, gamma, psi]
        us (array): Control inputs (e.g., bank angle)
        params (dict): Parameter dictionary with Mars-specific constants
        case_flag (int, optional): Determines how alpha is set (default: from params)

    Returns:
        dict: Containing arrays for 'L', 'D', 'Cl', 'Cd', 'alpha', and 'rho' (density)
    """
    # Unpack params (support nested dict in 'problem' style)
    if 'params' in params:
        params = params['params']

    if case_flag is None:
        case_flag = params['case_flag']

    # Dimensional constants
    rho0 = params['rhoe']           # Surface density
    H   = params['H']               # Scale height
    bc  = params['bc']              # Ballistic coefficient (m / (Cd*A))
    LD  = params.get('LD', None)    # Lift-to-drag ratio, optional

    # Extract number of points
    N = 1 if np.isscalar(ts) else (ts.shape[0] if ts.ndim == 1 else ts.shape[1])

    # Storage
    L     = np.zeros(N)
    D     = np.zeros(N)
    alpha = np.zeros(N)
    rho   = np.zeros(N)

    for k in range(N):
        tk = ts if N == 1 else ts[k]
        zk = zs if N == 1 else zs[k]
        uk = us if N == 1 else us[k]

        r, theta, phi, v, gamma, psi = zk
        v2 = v**2
        
        # Atmospheric density
        h_alt = r - params['re']
        rho_k = rho0 * np.exp(-h_alt / H)

        # Compute Drag and Lift using L/D profile or Coeff-based
        if LD is not None:
            D_k = rho_k * v2 / (2 * bc)
            L_k = D_k * LD
        else:
            # Default: no lift
            D_k = rho_k * v2 / (2 * bc)
            L_k = 0.0

        # Bank angle handling
        if case_flag == 1:
            sigma = uk  # bank directly from control
        else:
            sigma = uk[0] if np.ndim(uk) > 0 else uk
        
        # Flight path angle determination
        alpha_k = 0.0  # No-angle-of-attack control for simple Mars entry

        # Store
        rho[k]   = rho_k
        D[k]     = D_k
        L[k]     = L_k
        alpha[k] = alpha_k

    return {
        'L': L,
        'D': D,
        'alpha': alpha,
        'rho': rho
    }


def system_dynamics(ts, zs, us, params, v_vec=None):
    """
    Mars 3DoF reentry dynamics in polar coordinates with rotating planet (if enabled).
    
    Parameters:
        ts      : float or array-like
        zs      : np.ndarray, state vector [r, theta, phi, v, gamma, psi]
        us      : np.ndarray, control input (e.g., bank angle σ)
        params  : dictionary of physical/scaling parameters
        v_vec   : time vector for interpolated control (optional)

    Returns:
        xDot    : np.ndarray, time derivative of the state vector
    """
    # Extract parameters
    re        = params['re']
    Om        = params['Omega_s']
    Kg        = params['kg']
    n         = int(params['n'])
    m         = int(params['m'])
    case_flag = params['case_flag']
    N         = extract_N(ts)

    # Extract current state
    r, theta, phi, v, gamma, psi = zs

    # Interpolate control if time vector is provided
    if v_vec is not None:
        us2 = np.zeros(m)
        for i in range(m):
            us2[i] = np.interp(ts, v_vec, us[:, i])
    else:
        us2 = reference_bank_angle(ts, zs, params) 

    # Bank angle σ
    sigma = us2[0] if hasattr(us2, '__len__') else us2

    # === Call aero model ===
    aero = nonlinear_aero(ts, zs, us2, params)
    L     = aero['L'][0]  # scalar
    D     = aero['D'][0]
    alpha = aero['alpha'][0]

    # === Call mass/thrust model ===
    force = mass_thrust(ts, zs, us2, params)
    mass  = force['mass']
    Tf    = force['Tf']

    # === Trig terms
    cp, sp = np.cos(phi), np.sin(phi)
    tp     = np.tan(phi)
    cg, sg = np.cos(gamma), np.sin(gamma)
    tg     = np.tan(gamma)
    cps, sps = np.cos(psi), np.sin(psi)
    cs, ss   = np.cos(sigma), np.sin(sigma)
    ca, sa   = np.cos(alpha), np.sin(alpha)

    # === Dynamics
    xDot = np.zeros(n)

    # r_dot
    xDot[0] = v * sg

    # theta_dot
    xDot[1] = v * cg * sps / (r * cp)

    # phi_dot
    xDot[2] = v * cg * cps / r

    # v_dot
    xDot[3] = (Tf / mass) * ca - D - Kg * sg / r**2 \
              + Om**2 * r * cp * (sg * cp - cg * sp * cps)

    # gamma_dot
    xDot[4] = (1 / v) * ( ((Tf / mass) * sa + L) * cs + (v**2 - Kg / r) * cg / r ) \
              + 2 * Om * cp * sps \
              + Om**2 * r * (1 / v) * cp * (cg * cp + sg * cps * sp)

    # psi_dot
    xDot[5] = (1 / v) * ( ((Tf / mass) * sa + L) * ss / cg + v**2 * cg * sps * tp / r ) \
              - 2 * Om * (tg * cps * cp - sp) \
              + Om**2 * r * (1 / (v * cg)) * sps * sp * cp

    # Optional: m_dot if case_flag == 3
    if case_flag == 3:
        ce = params['ce']
        xDot = np.append(xDot, ce * Tf)  # xDot[6] = m_dot

    # Safety check
    if isinstance(r, (int, float)) and r <= 1:
        xDot = np.zeros(n)
    elif np.any(np.isnan(xDot)) or np.any(np.isinf(xDot)):
        raise ValueError("NaN or Inf values encountered in xDot")

    return xDot

# Example usage
if __name__ == "__main__":
    print("This module provides functions for reentry 3DoF dynamics and aerodynamics.")