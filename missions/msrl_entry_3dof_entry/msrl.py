import jax.numpy as jnp
from trajopt.utils.tools import AttrDict

# x = [r, theta, phi, v, fpa, heading]
# u = [bank, aoa]

def density_model(t, x, u, params, fcns):

    r = x[0]

    y = (r / (3.396 * 10**6)) - 1

    a = -4.418967
    b = -56.76770
    c = 1.621266
    d = 379.3357
    e = 8564.256
    f = 8414.405
    g = -988.7042
    h = 4461.988
    i = 31.36062

    numerator   = a + c*y + e*y**2 + g*y**3 + i*y**4
    denominator = 1 + b*y + d*y**2 + f*y**3 + h*y**4

    rho = jnp.exp(numerator / denominator)

    return rho

def nonlinear_aero(t, x, u, params, fcns):

    v = x[3]

    y = v / 3554.6731

    a_l = 0.679945
    b_l = - 9.522473
    c_l = 59.179219
    d_l = - 215.423516
    e_l = 510.853276
    f_l = -830.748818
    g_l = 950.127376
    h_l = -771.365024
    i_l = 443.191605
    j_l = -177.601759
    k_l = 48.232018
    l_l = -8.362472
    m_l = 1.177526

    Cl = (
        a_l*y**12 + b_l*y**11 + c_l*y**10 + d_l*y**9 + e_l*y**8 
        + f_l*y**7 + g_l*y**6 + h_l*y**5 + i_l*y**4 + j_l*y**3 
        + k_l*y**2 + l_l*y + m_l
    )

    a_d = 1.458554
    b_d = -20.501723
    c_d = 128.239298
    d_d = -470.835046
    e_d = 1126.686522
    f_d = -1844.173832
    g_d = 2106.728736
    h_d = -1681.164677
    i_d = 921.349863
    j_d = -333.513962
    k_d = 74.164741
    l_d = - 8.758177
    m_d = 1.684456

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