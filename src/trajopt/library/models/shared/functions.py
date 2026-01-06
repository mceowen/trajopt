import jax 
import jax.numpy as jnp
import cvxpy as cp

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




