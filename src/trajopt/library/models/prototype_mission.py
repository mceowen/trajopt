"""
VTOL Reentry Mission - Python Module Format
"""

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import trajopt.library.models.reentry_3dof.prototype_model as model


# =============================================================================
# PLANET DATA
# =============================================================================

planet = {
    'g': 9.81,
    'r': 6378137.0,
    'rho': 1.3,
    'H': 7000.0,
    'mu': 398600441800000.0,
    'omega': 0.004178,  # deg/s
}


# =============================================================================
# VEHICLE DATA
# =============================================================================

vehicle = {
    'mass': 104305.0,
    'sref': 391.22,
    'ce': 0.5,
    'kQ': 1.2035e-5,
}


# =============================================================================
# FLAGS
# =============================================================================

flags = {
    'flag_nfz': 0,
    'init_ctrl': 0,
    'final_ctrl': 0,
    'aero_type': 'exponential',
}


# =============================================================================
# MISSION FUNCTIONS
# =============================================================================

def atmosphere_model_jax(r, params):
    h = r - params['mission']['planet']['r']
    if params['mission']['flags']['aero_type'] == "exponential":
        rho = params['mission']['planet']['rho'] * jnp.exp(-h / params['mission']['planet']['H'])
    return rho


def atmosphere_model_nonjax(r, params):
    h = r - params['mission']['planet']['r']
    if params['mission']['flags']['aero_type'] == "exponential":
        rho = params['mission']['planet']['rho'] * np.exp(-h / params['mission']['planet']['H'])
    return rho


def nonlinear_aero_jax(r, v, params):
    ctrl_type = params['model']['flags']['ctrl_type']

    kl1, kl2, kl3 = -0.041065, 0.016292, 0.0002602
    kd1, kd2, kd3 = 0.080505, -0.03026, 0.86495
    kalph = 0.20705 / (340**2)
    vlim, alphlim_deg = 4570, 40

    if ctrl_type == 'bank_only':
        Kd1, Kd2, Kd3 = kd1, kd2, kd3
        Kl1 = kl1 + kl2 * alphlim_deg + kl3 * alphlim_deg**2
        Kl2 = -kl2 * kalph - 2 * kl3 * alphlim_deg * kalph
        Kl3 = kl3 * kalph**2

    v_sat = jnp.minimum(v, vlim)
    Cl = Kl1 + Kl2 * (v_sat - vlim)**2 + Kl3 * (v_sat - vlim)**4
    Cd = Kd1 + Kd2 * Cl + Kd3 * Cl**2

    rho = atmosphere_model_jax(r, params)
    sref = params['mission']['vehicle']['sref']
    mass = params['mission']['vehicle']['mass']

    L = (0.5 / mass) * rho * sref * Cl * v**2
    D = (0.5 / mass) * rho * sref * Cd * v**2

    return {'L': L, 'D': D, 'Cl': Cl, 'Cd': Cd, 'rho': rho}


def nonlinear_aero_nonjax(r, v, params):
    ctrl_type = params['model']['flags']['ctrl_type']

    kl1, kl2, kl3 = -0.041065, 0.016292, 0.0002602
    kd1, kd2, kd3 = 0.080505, -0.03026, 0.86495
    kalph = 0.20705 / (340**2)
    vlim, alphlim_deg = 4570, 40

    if ctrl_type == 'bank_only':
        Kd1, Kd2, Kd3 = kd1, kd2, kd3
        Kl1 = kl1 + kl2 * alphlim_deg + kl3 * alphlim_deg**2
        Kl2 = -kl2 * kalph - 2 * kl3 * alphlim_deg * kalph
        Kl3 = kl3 * kalph**2

    v_sat = np.minimum(v, vlim)
    Cl = Kl1 + Kl2 * (v_sat - vlim)**2 + Kl3 * (v_sat - vlim)**4
    Cd = Kd1 + Kd2 * Cl + Kd3 * Cl**2

    rho = atmosphere_model_nonjax(r, params)
    sref = params['mission']['vehicle']['sref']
    mass = params['mission']['vehicle']['mass']

    L = (0.5 / mass) * rho * sref * Cl * v**2
    D = (0.5 / mass) * rho * sref * Cd * v**2

    return {'L': L, 'D': D, 'Cl': Cl, 'Cd': Cd, 'rho': rho}


fcns = {
    'atmosphere_model_jax': atmosphere_model_jax,
    'atmosphere_model_nonjax': atmosphere_model_nonjax,
    'nonlinear_aero_jax': nonlinear_aero_jax,
    'nonlinear_aero_nonjax': nonlinear_aero_nonjax,
}


# =============================================================================
# CONSTRAINT PARAMETERS (add mission-specific values to model templates)
# =============================================================================

constraint_models = model.constraint_models

constraint_models['initial_state'].addParams(
    x=np.array([planet['r'] + 100e3, 0.0, 0.0, 7450, -0.5, 0.0]),
    x_idx=[0, 1, 2, 3, 4, 5],
)

constraint_models['final_state'].addParams(
    x=np.array([planet['r'] + 15e3, 12, 70, -10, 90]),
    x_idx=[0, 1, 2, 4, 5],
    eps=[100, 0.1, 0.1, 0.1, 0.1],
)

constraint_models['state_limits'].addParams(
    x_min=np.array([planet['r'], 100.0]),
    x_min_idx=[0, 3],
)

constraint_models['bank_angle_limit'].addParams(
    x_min=[-80.0],
    x_min_idx=[0],
    x_max=[80.0],
    x_max_idx=[0],
)

constraint_models['bank_angle_rate_limit'].addParams(
    udot_max=[10],
    udot_max_idx=[0],
)

constraint_models['max_Q'].addParams(max_val=33300, eps=[100])
constraint_models['max_q'].addParams(max_val=18000, eps=[180])
constraint_models['max_load'].addParams(max_val=24.525, eps=[0.01])


# =============================================================================
# ACTIVE COSTS
# =============================================================================

cost_models = model.cost_models

costs = ['min_terminal_velocity']