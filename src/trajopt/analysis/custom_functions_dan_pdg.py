import numpy as np
import jax 
import jax.numpy as jnp
import trajopt.core.modules.utils.tools as tools
jax.config.update("jax_enable_x64", True)
import trajopt.core.modules.model.obstacles     as obstacles
# import trajopt.local.modules.mission.aero.cobra_aero_nonjax as aero_nonjax



# Direction Cosine Matrix Function
def DCM(q): 
    return np.array(
        [
            [
                1 - 2 * (q[2] ** 2 + q[3] ** 2),
                2 * (q[1] * q[2] + q[0] * q[3]),
                2 * (q[1] * q[3] - q[0] * q[2]),
            ],
            [
                2 * (q[1] * q[2] - q[0] * q[3]),
                1 - 2 * (q[1] ** 2 + q[3] ** 2),
                2 * (q[2] * q[3] + q[0] * q[1]),
            ],
            [
                2 * (q[1] * q[3] + q[0] * q[2]),
                2 * (q[2] * q[3] - q[0] * q[1]),
                1 - 2 * (q[1] ** 2 + q[2] ** 2),
            ],
        ]
    )

def calc_DCMs(t, z, nu, trajopt_obj):
    veh = trajopt_obj.mission.vehicle
    rt = np.array([veh["rt1"], veh["rt2"], veh["rt3"]])
    dcm = DCM(z[7:11])
    return dcm 

def calc_rt_I(t, z, nu, trajopt_obj):
    veh = trajopt_obj.mission.vehicle
    rt = np.array([veh["rt1"], veh["rt2"], veh["rt3"]])
    dcm = DCM(z[7:11])
    return dcm.T @ rt

def thrust_mag(t,z,nu,trajopt_obj):
    return np.linalg.norm(nu[:trajopt_obj.model.m]);

def ang_rate(t,z,nu,trajopt_obj):
    return (180/np.pi)*np.linalg.norm(z[11:14]);

def compute_tilt(t,z,nu,trajopt_obj):
    return (180/np.pi)*np.arccos(1 - 2*(z[9]**2 + z[10]**2))

def omega_degrees(t,z,nu,trajopt_obj):
    return (180/np.pi)*z[11:14]


scale2d = 0.2
def calc_u_vecs_scale1(t, z, nu, trajopt_obj):
    veh = trajopt_obj.mission.vehicle
    rt = np.array([veh["rt1"], veh["rt2"], veh["rt3"]])
    dcm = DCM(z[7:11])
    return -0.3* dcm.T @ nu[:trajopt_obj.model.m]

def calc_u_vecs_scale2(t, z, nu, trajopt_obj):
    veh = trajopt_obj.mission.vehicle
    rt = np.array([veh["rt1"], veh["rt2"], veh["rt3"]])
    dcm = DCM(z[7:11])
    return -0.3*scale2d* dcm.T @ nu[:trajopt_obj.model.m]


def calc_body_vecs_scale1(t, z, nu, trajopt_obj):
    veh = trajopt_obj.mission.vehicle
    rt = np.array([veh["rt1"], veh["rt2"], veh["rt3"]])
    dcm = DCM(z[7:11])
    return 0.5 * dcm.T @ np.array([1, 0, 0]);

def calc_body_vecs_scale2(t, z, nu, trajopt_obj):
    veh = trajopt_obj.mission.vehicle
    rt = np.array([veh["rt1"], veh["rt2"], veh["rt3"]])
    dcm = DCM(z[7:11])
    return 0.5*scale2d * dcm.T @ np.array([1, 0, 0]);




def compute_altitude(t, z, nu, trajopt_obj):  #dynamic pressure
    mission = trajopt_obj.mission; #method = trajopt_obj.method
    return (z[0] - mission.planet['r'])/1000;

def max_q_nonjax(t, z, nu, trajopt_obj):  #dynamic pressure
    mission = trajopt_obj.mission; method = trajopt_obj.method
    rs = z[0]; vs = z[3]
    rho = atmosphere_model_nonjax(rs/method.nondim['nd'], trajopt_obj)
    nv = 1; #method.nondim["nv"]
    q_dim = 0.5 * rho * (vs * nv) ** 2 
    return q_dim #np.array([q_dim / mission.path_limits['max_q'] - 1.0])

def max_Q_nonjax(t, z, nu, trajopt_obj): # heat rate
    mission = trajopt_obj.mission; method = trajopt_obj.method
    rs = z[0]; vs = z[3]
    rho = atmosphere_model_nonjax(rs/method.nondim['nd'], trajopt_obj)
    nv = 1; #method.nondim["nv"]
    Q_dim = mission.vehicle["kQ"] * rho ** 0.5 * (vs * nv) ** 3
    return Q_dim #np.array([Q_dim / mission.path_limits["max_Q"] - 1.0])

def max_load_nonjax(t, z, nu, trajopt_obj): # normal load
    mission = trajopt_obj.mission; method = trajopt_obj.method
    rs = z[0]; vs = z[3]
    Mstate = method.nondim['M']['state']['d2nd']
    stateidx = trajopt_obj.indices.z['state']
    Mctrl = method.nondim['M']['ctrl']['d2nd']
    aero = nonlinear_aero_nonjax(t, z[stateidx] @ Mstate, nu @ Mctrl,trajopt_obj)
    # aero = nonlinear_aero_nonjax(t, z, nu,trajopt_obj)
    L = aero["L"]; D = aero["D"]
    load_dim = np.sqrt(L ** 2 + D ** 2) * method.nondim["na"] / mission.planet['g']; # (CARLOS): IM DIVIDING BY G TO GET LOAD IN G'S FOR THE PLOTS
    return load_dim #np.array([load_dim / mission.path_limits["max_load"] - 1.0])


# import numpy as np
# import trajopt.core.modules.utils.tools as tools

# import numpy as np
# import jax 
# import jax.numpy as jnp
# import trajopt.core.modules.utils.tools as tools
# jax.config.update("jax_enable_x64", True)

# =============================== LOCAL AERO DATA: ===============================
# 'lookup' option for nonlinear_aero relies on local aero data not in repo

# import sys
# sys.path.append("/Users/carlosm/Documents/guidance/hypersonics/prototypes/local")
# import marsgram_dens_lut as dens
# ================================================================================

def terminal_cost(t, z, nu, trajopt_obj):
    return z[3]

def atmosphere_model_nonjax(rs, trajopt_obj):
    '''
    Returns density as a function of orbital radius

    TODO: (carlos)
    expand 'lookup' option to be higher dimensional
    remember LUT grow exponenetiall in size!
    '''

    mission = trajopt_obj.mission
    model = trajopt_obj.model
    method = trajopt_obj.method

    rdim = rs * method.nondim["nd"]
    hdim = rdim - mission.planet["r"]

    # TODO (carlos): add the remaining options for atmosphere model
    if mission.flags["aero_type"] == "lookup":
        rho = np.interp(hdim/1e3, dens.h_grid, dens.rho_vals)

    elif mission.flags["aero_type"] == "exponential":
        rho = mission.planet["rho"] * np.exp(-hdim / mission.planet["H"])

    return rho

# Atmospheric temperature model
def mars_temperature_nonjax(hdim):

    T_0 = 249.75     # K (surface temperature on Mars), 210
    L   = 0.0025    # K/m (lapse rate), .0025
    T = np.maximum(T_0 - L * hdim, 150.0)
    
    return T  # Ensure temperature doesn't fall below ~150K

# Speed of sound calculation on Mars
def mars_speed_of_sound_nonjax(hdim):
    T = mars_temperature_nonjax(hdim)
    gamma_M = 1.29
    Rgas_M  = 188.92 
    return np.sqrt(gamma_M * Rgas_M * T)

def nonlinear_aero_nonjax(ts, zs, us, trajopt_obj):

    mission = trajopt_obj.mission
    model   = trajopt_obj.model
    method  = trajopt_obj.method

    # Extract states and controls

    rs = zs[0]
    vs = zs[3]

    alpha = us[1]
    alpha_deg = np.rad2deg(alpha)
    mass_s = mission.vehicle['mass'] / method.nondim['nm']

    rdim = rs * method.nondim["nd"]
    hdim = rdim - mission.planet["r"]

    rho = atmosphere_model_nonjax(rs, trajopt_obj)
    a = mars_speed_of_sound_nonjax(hdim)

    M = vs / (a / method.nondim['nv'])

    Cd, Cl = aero_nonjax.get_cd_cl(M, alpha_deg)

    rho_s = rho / (method.nondim['nm'] / method.nondim['nd'] ** 3)
    sref_s = mission.vehicle['sref'] / method.nondim['nd'] ** 2

    L    = (1 / mass_s) * 0.5 * rho_s * vs**2 * Cl * sref_s
    D    = (1 / mass_s) * 0.5 * rho_s * vs**2 * Cd * sref_s

    return {'L': L, 'D': D, 'Cl': Cl, 'Cd': Cd, 'alpha': alpha,'rho': rho}