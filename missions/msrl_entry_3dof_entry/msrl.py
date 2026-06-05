import jax.numpy as jnp
from trajopt.utils.tools import AttrDict

# x = [r, theta, phi, v, fpa, heading]
# u = [bank, aoa]

def density_model(t, x, u, params, fcns):

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

def nonlinear_aero(t, x, u, params, fcns):

    v = x[3]

    y = v / 3554.6731

    

    Cl = (
        a_l*y**12 + b_l*y**11 + c_l*y**10 + d_l*y**9 + e_l*y**8 
        + f_l*y**7 + g_l*y**6 + h_l*y**5 + i_l*y**4 + j_l*y**3 
        + k_l*y**2 + l_l*y + m_l
    )

    a_d = 
    b_d = 
    c_d = 
    d_d = -
    e_d = 1
    f_d = 
    g_d = 
    h_d = 
    i_d = 
    j_d = 
    k_d = 
    l_d =  
    m_d = 

    Cd = (
        a_d*y**12 + b_d*y**11 + c_d*y**10 + d_d*y**9 + e_d*y**8
        + f_d*y**7 + g_d*y**6 + h_d*y**5 + i_d*y**4 + j_d*y**3
        + k_d*y**2 + l_d*y + m_d
        )

    rho = fcns.density_model(t, x, u, params, fcns)

    mass = params.vehicle.mass
    sref = params.vehicle.sref

    L = (1 / mass) * 0.5 * rho * v**2 * Cl * sref
    D = (1 / mass) * 0.5 * rho * v**2 * Cd * sref

    return AttrDict({"L": L, "D": D, "Cl": Cl, "Cd": Cd})