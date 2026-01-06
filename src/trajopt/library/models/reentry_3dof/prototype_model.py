"""
Reentry 3DOF Model - Python Module Format
"""

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import trajopt.core.constraints.constraints_library as cnstrLib
import trajopt.core.costs.costs_library as costsLib


# =============================================================================
# DIMENSIONS
# =============================================================================

n = 6  # state dimension: [r, theta, phi, v, gamma, psi]
m = 2  # control dimension: [sigma (bank), alpha (optional)]


# =============================================================================
# NONDIMENSIONALIZATION
# =============================================================================

z_types = ["d", "ang", "ang", "v", "ang", "ang"]
u_types = ["ang", "ang"]
anchor_types = ["t", "v", "m"]

def anchor_scales(params):
    r = params['mission']['planet']['r']
    g = params['mission']['planet']['g']
    mass = params['mission']['vehicle']['mass']
    return np.array([(r/g)**0.5, (r*g)**0.5, mass])


# =============================================================================
# FLAGS
# =============================================================================

flags = {
    'ctrl_type': 'bank_only',
}


# =============================================================================
# DYNAMICS FUNCTION
# =============================================================================

def dynamics(t, z, nu, params, fcns):
    """3DOF reentry dynamics over a rotating spherical planet."""
    
    Om = jnp.deg2rad(params['mission']['planet']['omega'])
    mu = params['mission']['planet']['mu']

    r, theta, phi, v, gamma, psi = z
    sigma, alpha = nu[0], nu[1]

    phi_rad = jnp.deg2rad(phi)
    gamma_rad = jnp.deg2rad(gamma)
    psi_rad = jnp.deg2rad(psi)
    sigma_rad = jnp.deg2rad(sigma)

    aero = fcns['nonlinear_aero_jax'](r, v, params)
    L, D = aero["L"], aero["D"]

    cp, sp, tp = jnp.cos(phi_rad), jnp.sin(phi_rad), jnp.tan(phi_rad)
    cg, sg, tg = jnp.cos(gamma_rad), jnp.sin(gamma_rad), jnp.tan(gamma_rad)
    cps, sps = jnp.cos(psi_rad), jnp.sin(psi_rad)
    cs, ss = jnp.cos(sigma_rad), jnp.sin(sigma_rad)
    
    xDot = jnp.array([
        v * sg,
        jnp.rad2deg(v * cg * sps / (r * cp)),
        jnp.rad2deg(v * cg * cps / r), 
        -D - mu * sg / r**2 + Om**2 * r * cp * (sg * cp - cg * sp * cps),
        jnp.rad2deg((1/v) * (L * cs + (v**2 - mu/r) * cg/r) + 2*Om*cp*sps + Om**2*r*(1/v)*cp*(cg*cp + sg*cps*sp)),
        jnp.rad2deg((1/v) * (L * ss/cg + v**2*cg*sps*tp/r) - 2*Om*(tg*cps*cp - sp) + Om**2*r*(1/(v*cg))*sps*sp*cp)
    ])
    return xDot


# =============================================================================
# CONSTRAINT FUNCTIONS
# =============================================================================

def heat_rate(t, z, nu, params, fcns):
    r, v = z[0], z[3]
    rho = fcns['atmosphere_model_jax'](r, params)
    return jnp.array([params['mission']['vehicle']['kQ'] * rho**0.5 * v**3])


def dynamic_pressure(t, z, nu, params, fcns):
    r, v = z[0], z[3]
    rho = fcns['atmosphere_model_jax'](r, params)
    return jnp.array([0.5 * rho * v**2])


def aero_load(t, z, nu, params, fcns):
    r, v = z[0], z[3]
    aero = fcns['nonlinear_aero_jax'](r, v, params)
    L, D = aero["L"], aero["D"]
    return jnp.array([jnp.sqrt(L**2 + D**2)])


# =============================================================================
# CONSTRAINT MODELS
# =============================================================================

constraint_models = {

    'dynamics': cnstrLib.Dynamics(fcn=dynamics),

    'initial_state': cnstrLib.EqualityBC(set='state', boundary='init'),
    'final_state': cnstrLib.EqualityBC(set='state', boundary='final'),
    'final_bank_angle': cnstrLib.EqualityBC(set='control', boundary='final'),

    'state_limits': cnstrLib.Box(set='state'),
    'bank_angle_limit': cnstrLib.Box(set='control'),
    'bank_angle_rate_limit': cnstrLib.ControlRateLimit(),

    'max_Q': cnstrLib.NonconvexInequality(fcn=heat_rate, group='path', dimension=1),
    'max_q': cnstrLib.NonconvexInequality(fcn=dynamic_pressure, group='path', dimension=1),
    'max_load': cnstrLib.NonconvexInequality(fcn=aero_load, group='path', dimension=1),

}

# =============================================================================
# COST MODELS
# =============================================================================

cost_models = {

    'min_time': costsLib.MinTime(),
    'min_terminal_velocity': costsLib.TerminalState(x_idx=[3]),

}
