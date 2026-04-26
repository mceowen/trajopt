import jax
import jax.numpy as jnp
from jax import Array

jax.config.update("jax_enable_x64", True)

_d2r = jnp.pi / 180
_r2d = 180.0 / jnp.pi


def dynamics(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    """3-DoF RLV atmospheric entry dynamics (JAX)."""
    r, theta, phi, v, gamma, psi = z[0], z[1], z[2], z[3], z[4], z[5]
    aoa, bank = nu[0], nu[1]

    phi_r   = phi   * _d2r
    gamma_r = gamma * _d2r
    psi_r   = psi   * _d2r
    aoa_r   = aoa   * _d2r
    bank_r  = bank  * _d2r

    Re   = params["planet"]["r"]
    rho0 = params["planet"]["rho"]
    H    = params["planet"]["H"]
    mu   = params["planet"]["mu"]
    mass = params["vehicle"]["mass"]
    S    = params["vehicle"]["sref"]
    cl   = params["vehicle"]["cl"]
    cd   = params["vehicle"]["cd"]

    alt  = r - Re
    rho  = rho0 * jnp.exp(-alt / H)
    CL   = cl[0] + cl[1] * aoa_r
    CD   = cd[0] + cd[1] * aoa_r + cd[2] * aoa_r**2
    q    = 0.5 * rho * v**2
    L    = q * S * CL / mass
    D    = q * S * CD / mass
    grav = mu / r**2

    rdot     = v * jnp.sin(gamma_r)
    thetadot = v * jnp.cos(gamma_r) * jnp.sin(psi_r) / (r * jnp.cos(phi_r)) * _r2d
    phidot   = v * jnp.cos(gamma_r) * jnp.cos(psi_r) / r * _r2d
    vdot     = -D - grav * jnp.sin(gamma_r)
    gammadot = (L * jnp.cos(bank_r) - jnp.cos(gamma_r) * (grav - v**2 / r)) / v * _r2d
    psidot   = (L * jnp.sin(bank_r) / (v * jnp.cos(gamma_r))
                + v * jnp.cos(gamma_r) * jnp.sin(psi_r) * jnp.tan(phi_r) / r) * _r2d

    return jnp.array([rdot, thetadot, phidot, vdot, gammadot, psidot])


def altitude(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    return jnp.array([z[0] - params["planet"]["r"]])


def longitude(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    return jnp.array([z[1]])


def latitude(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    return jnp.array([z[2]])


def velocity(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    return jnp.array([z[3]])


def fpa(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    return jnp.array([z[4]])


def heading(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    return jnp.array([z[5]])


def aoa(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    return jnp.array([nu[0]])


def bank(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    return jnp.array([nu[1]])


def long_lat(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    return jnp.array([z[1], z[2]])


def heat_rate(t: float, z: Array, nu: Array, params: dict, fcns: dict) -> Array:
    r, v = z[0], z[3]
    aoa_deg = nu[0]
    Re   = params["planet"]["r"]
    rho0 = params["planet"]["rho"]
    H    = params["planet"]["H"]
    cl   = params["vehicle"]["cl"]
    cd   = params["vehicle"]["cd"]
    mass = params["vehicle"]["mass"]
    S    = params["vehicle"]["sref"]

    alt  = r - Re
    rho  = rho0 * jnp.exp(-alt / H)
    aoa_r = aoa_deg * _d2r
    CL   = cl[0] + cl[1] * aoa_r
    CD   = cd[0] + cd[1] * aoa_r + cd[2] * aoa_r**2
    q    = 0.5 * rho * v**2
    L    = q * S * CL / mass
    D    = q * S * CD / mass
    return jnp.array([jnp.sqrt(L**2 + D**2)])
