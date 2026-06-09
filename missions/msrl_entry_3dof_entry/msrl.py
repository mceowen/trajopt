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