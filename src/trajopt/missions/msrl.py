import jax.numpy as jnp
from trajopt.utils.tools import AttrDict

# x = [r, theta, phi, v, fpa, heading]
# u = [bank, aoa]

def density_model(x, u, t, params, fcns):

    r = x[0]

    y = (r / (3.396 * 10**6)) - 1

    a = params.a
    b = params.b
    c = params.c
    d = params.d
    e = params.e
    f = params.f
    g = params.g
    h = params.h
    i = params.i

    numerator   = a + c*y + e*y**2 + g*y**3 + i*y**4
    denominator = 1 + b*y + d*y**2 + f*y**3 + h*y**4

    rho = jnp.exp(numerator / denominator)

    return rho

def nonlinear_aero(x, u, t, params, fcns):

    v = x[3]

    y = v / 3554.6731

    a_l = params.a_l
    b_l = params.b_l
    c_l = params.c_l
    d_l = params.d_l
    e_l = params.e_l
    f_l = params.f_l
    g_l = params.g_l
    h_l = params.h_l
    i_l = params.i_l
    j_l = params.j_l
    k_l = params.k_l
    l_l = params.l_l
    m_l = params.m_l
    

    Cl = (
        a_l*y**12 + b_l*y**11 + c_l*y**10 + d_l*y**9 + e_l*y**8 
        + f_l*y**7 + g_l*y**6 + h_l*y**5 + i_l*y**4 + j_l*y**3 
        + k_l*y**2 + l_l*y + m_l
    )

    a_d = params.a_d
    b_d = params.b_d
    c_d = params.c_d
    d_d = params.d_d
    e_d = params.e_d
    f_d = params.f_d
    g_d = params.g_d
    h_d = params.h_d
    i_d = params.i_d
    j_d = params.j_d
    k_d = params.k_d
    l_d = params.l_d
    m_d = params.m_d

    Cd = (
        a_d*y**12 + b_d*y**11 + c_d*y**10 + d_d*y**9 + e_d*y**8
        + f_d*y**7 + g_d*y**6 + h_d*y**5 + i_d*y**4 + j_d*y**3
        + k_d*y**2 + l_d*y + m_d
        )

    rho = fcns.density_model(x, u, t, params, fcns)

    mass = params.vehicle.mass
    sref = params.vehicle.sref

    L = (1 / mass) * 0.5 * rho * v**2 * Cl * sref
    D = (1 / mass) * 0.5 * rho * v**2 * Cd * sref

    return AttrDict({"L": L, "D": D, "Cl": Cl, "Cd": Cd})

def parachute_aero(x, u, t, params, fcns):
    """Post-deployment drag acceleration, no lift from parachute."""
    v = x[3]

    rho = fcns.density_model(x, u, t, params, fcns)

    vehicle = params.vehicle
    cd_chute = vehicle.eta_chute * vehicle.cd_chute
    drag_area = vehicle.sref * vehicle.cd_backshell + vehicle.sref_chute * cd_chute

    D = (0.5 * rho * v**2 / vehicle.mass) * drag_area
    L = 0.0 * D

    return AttrDict({"L": L, "D": D})

def downrange_crossrange(x, u, t, params, fcns):
    """[downrange, crossrange] (m) to the touchdown target PCPF.

    Downrange is the great-circle range to the target resolved along the
    horizontal velocity direction, crossrange the component normal to it
    (positive when the target is right of track). Target longitude/latitude
    come from params.target (deg).
    """
    theta = jnp.deg2rad(x[1])
    phi = jnp.deg2rad(x[2])
    psi = jnp.deg2rad(x[5])

    theta_t = jnp.deg2rad(params.target.lon)
    phi_t = jnp.deg2rad(params.target.lat)

    # unit position vectors of vehicle and target in PCPF
    r_veh = jnp.array([
        jnp.cos(theta) * jnp.cos(phi),
        jnp.sin(theta) * jnp.cos(phi),
        jnp.sin(phi),
    ])
    r_tgt = jnp.array([
        jnp.cos(theta_t) * jnp.cos(phi_t),
        jnp.sin(theta_t) * jnp.cos(phi_t),
        jnp.sin(phi_t),
    ])

    # local North/East unit vectors at the vehicle in PCPF
    n_hat = jnp.array([
        -jnp.cos(theta) * jnp.sin(phi),
        -jnp.sin(theta) * jnp.sin(phi),
        jnp.cos(phi),
    ])
    e_hat = jnp.array([-jnp.sin(theta), jnp.cos(theta), 0.0])

    # bearing to target, clockwise from north
    bearing = jnp.arctan2(jnp.dot(r_tgt, e_hat), jnp.dot(r_tgt, n_hat))

    # atan2 form of the great-circle central angle rather than acos(dot)
    central = jnp.arctan2(
        jnp.linalg.norm(jnp.cross(r_veh, r_tgt)),
        jnp.dot(r_veh, r_tgt),
    )
    R = params.planet.r * central

    return jnp.array([R * jnp.cos(bearing - psi), R * jnp.sin(bearing - psi)])

def deploy_range_bias(x, u, t, params, fcns):
    """Deploy-surface residual [downrange - range_bias, crossrange] (m).

    Zero on the arc params.target.range_bias (m) uprange of the touchdown
    target with heading aligned toward it (deploy range bias, Mendeck & Craig
    AIAA 2011-6639, Fig. 3).
    """
    dr_cr = downrange_crossrange(x, u, t, params, fcns)
    return dr_cr - jnp.array([params.target.range_bias, 0.0])

def target_miss_distance_sq(x, u, t, params, fcns):
    """Squared great-circle miss distance (m^2) to the touchdown target."""
    theta = jnp.deg2rad(x[1])
    phi = jnp.deg2rad(x[2])

    theta_t = jnp.deg2rad(params.target.lon)
    phi_t = jnp.deg2rad(params.target.lat)

    r_veh = jnp.array([
        jnp.cos(theta) * jnp.cos(phi),
        jnp.sin(theta) * jnp.cos(phi),
        jnp.sin(phi),
    ])
    r_tgt = jnp.array([
        jnp.cos(theta_t) * jnp.cos(phi_t),
        jnp.sin(theta_t) * jnp.cos(phi_t),
        jnp.sin(phi_t),
    ])

    central = jnp.arctan2(
        jnp.linalg.norm(jnp.cross(r_veh, r_tgt)),
        jnp.dot(r_veh, r_tgt),
    )

    return jnp.array([(params.planet.r * central)**2])

def deploy_position_error_sq(x, u, t, params, fcns):
    """Squared local-horizontal position error (m^2) to params.deploy_target."""
    theta_d = jnp.deg2rad(params.deploy_target.lon)
    phi_d = jnp.deg2rad(params.deploy_target.lat)

    e_dr = params.planet.r * (jnp.deg2rad(x[1]) - theta_d) * jnp.cos(phi_d)
    e_cr = params.planet.r * (jnp.deg2rad(x[2]) - phi_d)

    return jnp.array([e_dr**2 + e_cr**2])
