import casadi as ca
import numpy as np

_d2r = np.pi / 180
_r2d = 180.0 / np.pi


def dynamics(t, z, nu, params, fcns):
    r, theta, phi, v, gamma, psi = z[0], z[1], z[2], z[3], z[4], z[5]
    aoa, bank = nu[0], nu[1]

    phi_r   = phi   * _d2r
    gamma_r = gamma * _d2r
    psi_r   = psi   * _d2r
    aoa_r   = aoa   * _d2r
    bank_r  = bank  * _d2r

    Re   = params.planet.r
    rho0 = params.planet.rho
    H    = params.planet.H
    mu   = params.planet.mu
    mass = params.vehicle.mass
    S    = params.vehicle.sref
    cl   = params.vehicle.cl
    cd   = params.vehicle.cd

    alt  = r - Re
    rho  = rho0 * ca.exp(-alt / H)
    CL   = cl[0] + cl[1] * aoa_r
    CD   = cd[0] + cd[1] * aoa_r + cd[2] * aoa_r**2
    q    = 0.5 * rho * v**2
    L    = q * S * CL / mass
    D    = q * S * CD / mass
    grav = mu / r**2

    rdot     = v * ca.sin(gamma_r)
    thetadot = v * ca.cos(gamma_r) * ca.sin(psi_r) / (r * ca.cos(phi_r)) * _r2d
    phidot   = v * ca.cos(gamma_r) * ca.cos(psi_r) / r * _r2d
    vdot     = -D - grav * ca.sin(gamma_r)
    gammadot = (L * ca.cos(bank_r) - ca.cos(gamma_r) * (grav - v**2 / r)) / v * _r2d
    psidot   = (L * ca.sin(bank_r) / (v * ca.cos(gamma_r))
                + v * ca.cos(gamma_r) * ca.sin(psi_r) * ca.tan(phi_r) / r) * _r2d

    return ca.vertcat(rdot, thetadot, phidot, vdot, gammadot, psidot)


def altitude(t, z, nu, params, fcns):
    return ca.vertcat(z[0] - params.planet.r)


def longitude(t, z, nu, params, fcns):
    return ca.vertcat(z[1])


def latitude(t, z, nu, params, fcns):
    return ca.vertcat(z[2])


def velocity(t, z, nu, params, fcns):
    return ca.vertcat(z[3])


def fpa(t, z, nu, params, fcns):
    return ca.vertcat(z[4])


def heading(t, z, nu, params, fcns):
    return ca.vertcat(z[5])


def aoa_out(t, z, nu, params, fcns):
    return ca.vertcat(nu[0])


def bank_out(t, z, nu, params, fcns):
    return ca.vertcat(nu[1])


def long_lat(t, z, nu, params, fcns):
    return ca.vertcat(z[1], z[2])


def heat_rate(t, z, nu, params, fcns):
    r, v = z[0], z[3]
    aoa_deg = nu[0]
    Re   = params.planet.r
    rho0 = params.planet.rho
    H    = params.planet.H
    cl   = params.vehicle.cl
    cd   = params.vehicle.cd
    mass = params.vehicle.mass
    S    = params.vehicle.sref

    alt  = r - Re
    rho  = rho0 * ca.exp(-alt / H)
    aoa_r = aoa_deg * _d2r
    CL   = cl[0] + cl[1] * aoa_r
    CD   = cd[0] + cd[1] * aoa_r + cd[2] * aoa_r**2
    q    = 0.5 * rho * v**2
    L    = q * S * CL / mass
    D    = q * S * CD / mass
    return ca.vertcat(ca.sqrt(L**2 + D**2))
