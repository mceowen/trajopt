import numpy as np
import jax 
import jax.numpy as jnp
import trajopt.utils.tools as tools
jax.config.update("jax_enable_x64", True)
import trajopt.core.modules.model.obstacles     as obstacles

def dynamics_jax(t, z, nu, trajopt_obj, t_vec=None):

    mission = trajopt_obj.mission
    model = trajopt_obj.model
    method = trajopt_obj.method

    ctrl_type = model.flags['ctrl_type']

    # Extract constant param values from struct
    Om = mission.planet["omega"] / (method.nondim["nang"] / method.nondim["nt"])
    Kg = mission.planet["mu"]    / (method.nondim["na"] * method.nondim["nd"] ** 2)

    # Extract states
    rs, theta, phi, vs, gamma, psi = z

    if ctrl_type == 'bank_aoa':
        alpha   = nu[1]
    
    sigma       = nu[0]
    
    # Determine lift and drag coefficients from velocity
    aero = mission.nonlinear_aero(t, z, nu)
    L    = aero["L"]
    D    = aero["D"]

    # Extract sines and cosines of various values
    cp  = jnp.cos(phi)
    sp  = jnp.sin(phi)
    tp  = jnp.tan(phi)
    cg  = jnp.cos(gamma)
    sg  = jnp.sin(gamma)
    tg  = jnp.tan(gamma)
    cps = jnp.cos(psi)
    sps = jnp.sin(psi)

    cs  = jnp.cos(sigma)
    ss  = jnp.sin(sigma)
    
    # state derivative function
    xDot = jnp.array([
        vs * sg,
        vs * cg * sps / (rs * cp),
        vs * cg * cps / rs, 
        - D - Kg * sg / rs**2 + Om**2 * rs * cp * (sg * cp - cg * sp * cps),
        (1 / vs) * ( L * cs + (vs**2 - Kg / rs) * cg / rs ) + 2 * Om * cp * sps + Om**2 * rs * (1 / vs) * cp * (cg * cp + sg * cps * sp),
        (1 / vs) * ( L * ss / cg + vs**2 * cg * sps * tp / rs ) - 2 * Om * (tg * cps * cp - sp) + Om**2 * rs * (1 / (vs * cg)) * sps * sp * cp
    ])

    return xDot

def max_q(t, z, nu, trajopt_obj):  #dynamic pressure
    mission = trajopt_obj.mission
    method = trajopt_obj.method
    
    rs = z[0]
    vs = z[3]

    rho = mission.mission_module.atmosphere_model_jax(rs, trajopt_obj)
    nv = method.nondim["nv"]

    q_dim = 0.5 * rho * (vs * nv) ** 2 

    return jnp.array([q_dim / mission.path_limits['max_q'] - 1.0])

def max_Q(t, z, nu, trajopt_obj): # heat rate
    mission = trajopt_obj.mission
    method = trajopt_obj.method

    rs = z[0]
    vs = z[3]

    rho = mission.mission_module.atmosphere_model_jax(rs, trajopt_obj)
    nv = method.nondim["nv"]

    Q_dim = mission.vehicle["kQ"] * rho ** 0.5 * (vs * nv) ** 3

    return jnp.array([Q_dim / mission.path_limits["max_Q"] - 1.0])

def max_load(t, z, nu, trajopt_obj): # normal load
    mission = trajopt_obj.mission
    method = trajopt_obj.method

    rs = z[0]
    vs = z[3]

    aero = mission.nonlinear_aero(t, z, nu)

    L = aero["L"]
    D = aero["D"]

    load_dim = jnp.sqrt(L ** 2 + D ** 2) * method.nondim["na"]

    return jnp.array([load_dim / mission.path_limits["max_load"] - 1.0])



def max_q_nonjax(t, z, nu, trajopt_obj):  #dynamic pressure
    mission = trajopt_obj.mission; method = trajopt_obj.method
    rs = z[0]; vs = z[3]
    rho = mission.mission_module.atmosphere_model_nonjax(rs, trajopt_obj)
    nv = method.nondim["nv"]
    q_dim = 0.5 * rho * (vs * nv) ** 2 
    return q_dim #np.array([q_dim / mission.path_limits['max_q'] - 1.0])

def max_Q_nonjax(t, z, nu, trajopt_obj): # heat rate
    mission = trajopt_obj.mission; method = trajopt_obj.method
    rs = z[0]; vs = z[3]
    rho = mission.mission_module.atmosphere_model_nonjax(rs, trajopt_obj)
    nv = method.nondim["nv"]
    Q_dim = mission.vehicle["kQ"] * rho ** 0.5 * (vs * nv) ** 3
    return Q_dim #np.array([Q_dim / mission.path_limits["max_Q"] - 1.0])

def max_load_nonjax(t, z, nu, trajopt_obj): # normal load
    mission = trajopt_obj.mission; method = trajopt_obj.method
    rs = z[0]; vs = z[3]
    aero = mission.nonlinear_aero_nonjax(t, z, nu)
    L = aero["L"]; D = aero["D"]
    load_dim = np.sqrt(L ** 2 + D ** 2) * method.nondim["na"]
    return load_dim #np.array([load_dim / mission.path_limits["max_load"] - 1.0])