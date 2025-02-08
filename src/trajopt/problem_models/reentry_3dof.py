def reentry_3dof_dynamics(ts, zs, us, params, t_vec=None):
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
    m           = params['m']
    case_flag   = params['case_flag']

    # Extract states
    r, theta, phi, v, gamma, psi = zs[:6]
    
    # Extract controls 
    if t_vec is None:
        us2 = us
    else:
        us2 = np.zeros_like(us)
        for i in range(m):
            us2[i, :] = np.interp(ts, t_vec, us[i, :])

    # Extract bank angle
    sigma = us2[0]

    # Determine lift and drag coefficients from velocity
    aero, alpha = nonlinear_aero(ts, zs, us2, params)
    L           = aero[0, 0]
    D           = aero[1, 0]
    Cl          = aero[2, 0]
    Cd          = aero[3, 0]

    # Extract mass and thrust
    mass, Tf = params['mass_thrust'](ts, zs, us2)

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
    xDot        = np.zeros((7, 1))
    
    # r_dot
    xDot[0, 0]  = v * sg
    # theta_dot
    xDot[1, 0]  = v * cg * sps / (r * cp)
    # phi_dot
    xDot[2, 0]  = v * cg * cps / r
    # v_dot
    xDot[3, 0]  = (Tf / mass) * ca - D - Kg * sg / r**2 + Om**2 * r * cp * (sg * cp - cg * sp * cps)
    # gamma_dot
    xDot[4, 0]  = (1 / v) * ((Tf / mass) * sa + L) * cs + (v**2 - Kg / r) * cg / r + 2 * Om * cp * sps + Om**2 * r * (1 / v) * cp * (cg * cp + sg * cps * sp)
    # psi_dot
    xDot[5, 0]  = (1 / v) * ((Tf / mass) * sa + L) * ss / cg + v**2 * cg * sps * tp / r - 2 * Om * (tg * cps * cp - sp) + Om**2 * r * (1 / (v * cg)) * sps * sp * cp

    if case_flag == 3:
        ce = params['ce']
        # m_dot
        xDot[6, 0] = ce * Tf

    if isinstance(r, (int, float)):
        if r <= 1:
            xDot = np.zeros((params['n'], 1))
    elif np.any(np.isnan(xDot)) or np.any(np.isinf(xDot)):
        raise ValueError("NaN or Inf values encountered in xDot")

    return xDot

# Example usage
if __name__ == "__main__":
    # Define dummy data for testing
    ts      = 0.0
    zs      = np.array([1, 0.5, 0.2, 3000, 0.05, 0.1, 1000])
    us      = np.array([[0.1], [0.2]])
    params  = {
        're': 6371e3,
        'rhoe': 1.225,
        'Omega_s': 7.2921159e-5,
        'kg': 3.986004418e14,
        'm': 2,
        'case_flag': 3,
        'ce': 0.5,
        'mass_thrust': lambda ts, zs, us2: (1000, 500)  # Dummy mass and thrust function
    }
    t_vec   = np.linspace(0, 10, 100)
    
    xDot    = reentry_3dof_dynamics(ts, zs, us, params, t_vec)
    print(xDot)