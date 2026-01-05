import numpy as np
import jax 
import jax.numpy as jnp
import trajopt.core.utils.tools  as tools
import trajopt.core.model.obstacles    as obstacles

#####################################################################
##################  ------- IN DEVELOPMENT ------- ##################
  
def qhatx(q,realloc=0):
    q1 = q[0]; q2 = q[1]; q3 = q[2]; q4 = q[3];
    if realloc == 0 or realloc == 1: ### not sure this is right
        out = np.array([[ q1, -q2,-q3,-q4 ],

                        [ q2,  q1,-q4, q3 ],
                        [ q3,  q4, q1,-q2 ],
                        [ q4, -q3, q2, q1 ]]);

    if realloc == 3 or realloc == 4:
        out = np.array([[ q4, q3,-q2, q1 ],
                        [-q3, q4, q1, q2 ],
                        [ q2,-q1, q4, q3 ],
                        [-q1,-q2,-q3, q4 ]]);
    return out

def qhato(q,realloc=3):
    q1 = q[0]; q2 = q[1]; q3 = q[2]; q4 = q[3]; 
    if realloc == 3 or realloc == 4:
        out = np.array([[ q4,-q3, q2, q1 ],
                        [ q3, q4,-q1, q2 ],
                        [-q2, q1, q4, q3 ],
                        [-q1,-q2,-q3, q4 ]]);
    return out

##############################################################################
################ ------------ READY TO GO -------------------#################
##############################################################################


def DCM(q):
    return np.array(
        [[1 - 2 * (q[2] ** 2 + q[3] ** 2),
          2 * (q[1] * q[2] + q[0] * q[3]),
          2 * (q[1] * q[3] - q[0] * q[2]),],
         [2 * (q[1] * q[2] - q[0] * q[3]),
          1 - 2 * (q[1] ** 2 + q[3] ** 2),
          2 * (q[2] * q[3] + q[0] * q[1]),],
         [2 * (q[1] * q[3] + q[0] * q[2]),
          2 * (q[2] * q[3] - q[0] * q[1]),
          1 - 2 * (q[1] ** 2 + q[2] ** 2),],])


# Direction Cosine Matrix Function
def jDCM(q): 
    return jnp.array(
        [[1 - 2 * (q[2] ** 2 + q[3] ** 2),
          2 * (q[1] * q[2] + q[0] * q[3]),
          2 * (q[1] * q[3] - q[0] * q[2]),],
         [2 * (q[1] * q[2] - q[0] * q[3]),
          1 - 2 * (q[1] ** 2 + q[3] ** 2),
          2 * (q[2] * q[3] + q[0] * q[1]),],
         [2 * (q[1] * q[3] + q[0] * q[2]),
          2 * (q[2] * q[3] - q[0] * q[1]),
          1 - 2 * (q[1] ** 2 + q[2] ** 2),],])    

# skew symmetric quaternion matrix
def omega(w):
    return np.array(
    [[0, -w[0], -w[1], -w[2]],
     [w[0], 0, w[2], -w[1]],
     [w[1], -w[2], 0, w[0]],
     [w[2], w[1], -w[0], 0],])

def jomega(w):
    return jnp.array(
    [[0, -w[0], -w[1], -w[2]],
     [w[0], 0, w[2], -w[1]],
     [w[1], -w[2], 0, w[0]],
     [w[2], w[1], -w[0], 0],])

# skew symmetric cross product matrix function
def cr(v): return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
def jcr(v): return jnp.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

# alternative formulation for DCM
# - much cleaner and more related to the Rodriguez formula which is clearly understandable
# - these check out with the formula above. 
def DCMb(q): return (q[0]*np.eye(3)-cr(q[1:]))@(q[0]*np.eye(3)-cr(q[1:]))+np.outer(q[1:],q[1:])
def jDCMb(q): return (q[0]*jnp.eye(3)-jcr(q[1:]))@(q[0]*np.eye(3)-jcr(q[1:]))+np.outer(q[1:],q[1:])





def dynamics(t, z, nu, trajopt_obj):
    """
    x1, x2: r (position)
    u1, u2: v (velocity)
    """
    # extracts params if 'trajopt_obj' parent struct is passed in
    mission = trajopt_obj.mission
    model = trajopt_obj.model
    method = trajopt_obj.method
    # extract constant param values
    m       = model.m; 
    n       = model.n
    mass    = mission.vehicle["mass"] / method.nondim["nm"]
    # J       = mission.vehicle["moment_of_inertia"]; 
    J = np.eye(3); Jinv = np.eye(3);
    g_vec   = np.array([0,0, -mission.planet["g"]]) / method.nondim["na"]
    ## extract states
    r = z[0:3]; v = z[3:6]; q = z[6:10]; omg = z[10:13];
    thrust = nu[:3]; torque = nu[3:];
    # compute velocity and acceleration
    xDot        = np.empty(13) # initialize
    xDot[0:3]   = v
    xDot[3:6]   = (1./mass)*DCM(q)@thrust + g_vec
    xDot[6:10]  = omega(omg)@q;
    xDot[10:13] = Jinv@torque
    return xDot

def jdynamics(t, z, nu, trajopt_obj):
    """
    x1, x2: r (position)
    u1, u2: v (velocity)
    """
    # extracts params if 'trajopt_obj' parent struct is passed in
    mission = trajopt_obj.mission
    model = trajopt_obj.model
    method = trajopt_obj.method
    # extract constant param values
    m       = model.m; 
    n       = model.n
    mass    = mission.vehicle["mass"] / method.nondim["nm"]
    # J       = mission.vehicle["moment_of_inertia"]; 
    J = jnp.eye(3); Jinv = jnp.eye(3);
    g_vec   = jnp.array([0,0, -mission.planet["g"]]) / method.nondim["na"]
    ## extract states
    r = z[0:3]; v = z[3:6]; q = z[6:10]; omg = z[10:13];
    thrust = nu[:3]; torque = nu[3:];
    # compute velocity and acceleration
    xDot        = jnp.empty(13) # initialize
    xDot[0:3]   = v
    xDot[3:6]   = (1./mass)*jDCM(q)@thrust + g_vec
    xDot[6:10]  = jomega(omg)@q;
    xDot[10:13] = Jinv@torque
    return xDot    

######################################################################
############## ----------- OLDER STUFF ------------- #################


def analytical_linsys(t, z, nu, trajopt_obj):

    mission = trajopt_obj.mission
    model = trajopt_obj.model
    method = trajopt_obj.method
    
    # Extract parameters

    n       = model.n
    m       = model.m
    mass    = mission.vehicle["mass"] / method.nondim["nm"]

    # Sanity check for vector shapes
    z =  np.asarray(z).flatten()
    nu = np.asarray(nu).flatten()

    assert len(z) == n, f"Expected state vector of length {n}, got {len(z)}"
    assert len(nu) == m, f"Expected control vector of length {m}, got {len(nu)}"

    # Compute A matrix (Jacobian w.r.t. state)
    n2 = n // 2
    Ac = np.block([
        [np.zeros((n2, n2)), np.eye(n2)],
        [np.zeros((n2, n))]
    ])

    # Compute B matrix (Jacobian w.r.t. control)
    Bc = np.vstack([
        np.zeros((n2, m)),
        np.eye(m)
    ]) * (1.0 / mass)

    # Evaluate nonlinear dynamics
    fc = dynamics(t, z, nu, trajopt_obj)

    return fc, Ac, Bc

def dynamics_jax(t, z, nu, trajopt_obj):
    mission = trajopt_obj.mission
    model = trajopt_obj.model
    method = trajopt_obj.method

    mass    = mission.vehicle["mass"] / method.nondim["nm"]
    g_vec   = np.array([0,0, -mission.planet["g"]]) / method.nondim["na"]

    r = z[0:3]
    v = z[3:6]
    
    T = nu

    xDot = jnp.concatenate([v, T/mass + g_vec])

    return xDot