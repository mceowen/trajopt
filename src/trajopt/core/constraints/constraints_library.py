import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp
import trajopt.core.scp.convexify as convexify
from trajopt.utils.config_loader import resolve_function

# NOTE TO SELF (CARLOS):
# let z = M_in @ x, then:
# g(z) <= max_val  <==>  M_out^-1 @ g(M_in @ x) <= M_out^-1 @ max_val

# GIVEN THAT M IS POSITIVE DIAGONAL!
# WHEN WE NONDIM WE ARE DOING BOTH SCALING OF INPUTS AND CONDITIONING OF OUTPUTS!
# only need linear scaling for now because we're using deviation variables in scp

# dynamics:
# let z = M_in_x @ x, and nu = M_in_u @ u
# z_dot = f(z, nu)
# M_in_x @ x_dot = f(M_in_x @ x, M_in_u @ u)
# dx/dt = M_in_x^-1 @ f(M_in_x @ x, M_in_u @ u) # WRT TO PHYSICAL TIME!
#
# WRT TO normalized time:
# let t = nt * tau, then:
# dx/dtau = nt * M_in_x^-1 @ f(M_in_x @ x, M_in_u @ u)

# local caobra example has better scaling 


# TODO: need to add affine approx of convex constraints for ctcs

# ===============================================================
# CONVEX CONSTRAINTS
# ===============================================================

class equality_bc:
    def __init__(self, name, set, x, x_idx, boundary, eps=None, params=None):

        # parameters
        self.name = name
        self.set = set
        self.x_dim = x
        self.x_idx = x_idx
        self.boundary = boundary
        self.idx = 0 if boundary == 'init' else -1 if boundary == 'final' else None
        if eps is not None:
            self.eps = eps
        else:
            self.eps = np.zeros(len(x_idx))
        self.dimension = len(x_idx)
        self.implement_type = 'equality_bc'


        self.x = None

    # written for nondim input
    def fcn(self,x):
        return x[self.x_idx] - self.x

    def nondim_constraint(self, nondim):
        if self.set == "state":
            self.x = nondim.M["state"]["d2nd"][np.ix_(self.x_idx, self.x_idx)] @ self.x_dim
        elif self.set == "control":
            self.x = nondim.M["ctrl"]["d2nd"][np.ix_(self.x_idx, self.x_idx)] @ self.x_dim

class inequality_bc:
    def __init__(self, name, set, x_min, x_min_idx, x_max, x_max_idx, boundary, eps=np.array([]), params=None):

        # parameters
        self.name = name
        self.set = set
        self.x_min_dim = x_min
        self.x_min_idx = x_min_idx
        self.x_max_dim = x_max
        self.x_max_idx = x_max_idx
        self.idx = 0 if boundary == 'init' else -1 if boundary == 'final' else None
        self.eps = eps
        self.dimension = len(x_min_idx) + len(x_max_idx)
        self.x_min = None
        self.x_max = None

        self.implement_type = 'inequality_bc'

    def nondim_constraint(self, nondim):
        if self.set == "state":
            self.x_min = nondim.M["state"]["d2nd"][np.ix_(self.x_min_idx, self.x_min_idx)] @ self.x_min_dim
            self.x_max = nondim.M["state"]["d2nd"][np.ix_(self.x_max_idx, self.x_max_idx)] @ self.x_max_dim
        elif self.set == "control":
            self.x_min = nondim.M["ctrl"]["d2nd"][np.ix_(self.x_min_idx, self.x_min_idx)] @ self.x_min_dim
            self.x_max = nondim.M["ctrl"]["d2nd"][np.ix_(self.x_max_idx, self.x_max_idx)] @ self.x_max_dim

# ---------------------------------------------------------------
# path inequality constraints
# ---------------------------------------------------------------
class box:
    def __init__(self, name, set, x_min, x_min_idx, x_max, x_max_idx, params=None):
        self.name = name
        self.set = set
        self.x_min_dim = x_min
        self.x_min_idx = x_min_idx
        self.x_max_dim = x_max
        self.x_max_idx = x_max_idx
        self.dimension = len(x_min_idx) + len(x_max_idx)

        self.x_min = None
        self.x_max = None

        self.implement_type = 'box'

        if self.set == "state":
            n_elem = params['model']['dimensions']['n']
        elif self.set == "control":
            n_elem = params['model']['dimensions']['m']

        M_min = -np.eye(n_elem)[self.x_min_idx, :]
        M_max = np.eye(n_elem)[self.x_max_idx, :]

        self.M_select = np.vstack([M_min, M_max])

    def nondim_constraint(self, nondim):
        if self.set == "state":
            self.x_min = nondim.M["state"]["d2nd"][np.ix_(self.x_min_idx, self.x_min_idx)] @ self.x_min_dim
            self.x_max = nondim.M["state"]["d2nd"][np.ix_(self.x_max_idx, self.x_max_idx)] @ self.x_max_dim

        if self.set == "control":
            self.x_min = nondim.M["ctrl"]["d2nd"][np.ix_(self.x_min_idx, self.x_min_idx)] @ self.x_min_dim
            self.x_max = nondim.M["ctrl"]["d2nd"][np.ix_(self.x_max_idx, self.x_max_idx)] @ self.x_max_dim

# ---------------------------------------------------------------
# rate constraints
# ---------------------------------------------------------------
class control_rate_limit:
    def __init__(self, name, udot_max, udot_max_idx, params=None):
        self.name = name
        self.udot_max_dim = udot_max
        self.udot_max_idx = udot_max_idx
        self.dimension = len(udot_max_idx)

        self.implement_type = 'control_rate_limit'

        n_elem = params['model']['dimensions']['m']
        M_min = -np.eye(n_elem)[self.udot_max_idx, :]
        M_max = np.eye(n_elem)[self.udot_max_idx, :]

        self.udot_max = None

        self.M_select = np.vstack([M_min, M_max])

    def nondim_constraint(self, nondim):
        self.udot_max = nondim.nt * nondim.M["ctrl"]["d2nd"][np.ix_(self.udot_max_idx, self.udot_max_idx)] @ self.udot_max_dim

# ---------------------------------------------------------------
# Second-order cone cosntraints
# ---------------------------------------------------------------

class axis_angle_cone:
    def __init__(self, name, set, axis, theta_max, x_idx, params=None):
        self.name = name
        self.set = set
        self.axis = axis / np.linalg.norm(axis)
        self.cos_theta_max = np.cos(np.deg2rad(theta_max))
        self.x_idx = x_idx
        self.dimension = 1
        self.implement_type = 'axis_angle_cone'

    def nondim_constraint(self, nondim):
        # the deg2rad is already nondimming
        pass

class max_norm_cone:
    def __init__(self, name, set, max_val, x_idx, params=None):
        self.name = name
        self.set = set
        self.max_val_dim = max_val
        self.x_idx = x_idx
        self.dimension = 1
        self.implement_type = 'max_norm_cone'

        self.max_val = None

    def nondim_constraint(self, nondim):

        if self.set == "state":
            nondim_key = nondim.z_types[self.x_idx[0]]
            self.max_val = self.max_val_dim * nondim.scales[nondim_key]
        
        elif self.set == "control":
            nondim_key = nondim.u_types[self.x_idx[0]]
            self.max_val =  self.max_val_dim * nondim.scales[nondim_key]

class quaternion_cone:
    def __init__(self, name, theta_max, axis_num, quat_start_idx, params=None):
        self.name = name
        self.quat_start_idx = quat_start_idx
        self.cos_theta_max = np.cos(np.deg2rad(theta_max))
        self.axis_num = axis_num
        self.rhs = np.sqrt((1.0 - self.cos_theta_max) * 0.5)
        self.dimension = 1

        self.implement_type = 'quaternion_cone'

    def nondim_constraint(self, nondim):
        pass

# ===============================================================
# NONCONVEX CONSTRAINTS
# ===============================================================

# TODO: change to (func - max_val) / scale
class nonconvex_inequality:
    def __init__(self, name, group, fcn, units, dimension, ct, eps=None, max_val=None, mission_params=None, params=None):
        self.name = name
        self.group = group
        self.mission_params = mission_params
        self.params = params
        self.units = units
        self.eps = eps
        self.dimension = dimension
        self.ct = ct

        self.implement_type = 'nonconvex_inequality'
        self.fcn_dim = resolve_function(fcn)
        
        # this will be a function f(t, z, nu)
        self.fcn = None
        
        self.fcn_jit = None
        self.dfcn_dz_jit = None
        self.dfcn_du_jit = None

        self.has_max_val = max_val is not None
        
        if self.has_max_val is False:
            self.max_val_dim = None
        else:
            self.max_val_dim = jnp.atleast_1d(jnp.asarray(max_val))
    
    def g_aff(self, t, z, nu):
        return self.fcn_jit(z, nu), self.dfcn_dz_jit(z, nu), self.dfcn_du_jit(z, nu)

    def nondim_constraint(self, nondim):

        if self.max_val_dim is not None:
            M_out_d2nd = np.diag(1 / self.max_val_dim)
        else:
            M_out_d2nd, M_out_nd2d = nondim.build_nondim_matrix(self.units)

        if self.has_max_val:
            self.max_val = M_out_d2nd @ self.max_val_dim

        # fcn_dim is already bound with params/fcns by resolve_functions
        nd_fcn_lhs = nondim.nondim_function(self.fcn_dim, nondim.M["state"]["nd2d"], nondim.M["ctrl"]["nd2d"], M_out_d2nd)

        if self.has_max_val:
            self.fcn = lambda t, z, nu: nd_fcn_lhs(t, z, nu) - self.max_val
        else:
            self.fcn = lambda t, z, nu: nd_fcn_lhs(t, z, nu)

class dynamics:
    def __init__(self, name, fcn, mission_params=None, params=None):
        self.name = name
        self.mission_params = mission_params
        self.params = params
        
        # dynamically loaded from user module, gets closed
        # over with params and fcns dicts if necessary
        self.fcn_dim = resolve_function(fcn)

        self.implement_type = 'dynamics'

        # this will be a function f(t, z, nu) and operates on nondim variables
        self.fcn = None

        self.fcn_jit = None
        self.dfcn_dz_jit = None
        self.dfcn_du_jit = None

    def lin_dyn(self, t, z, nu):
        return self.fcn_jit(z, nu), self.dfcn_dz_jit(z, nu), self.dfcn_du_jit(z, nu)

    def nondim_constraint(self, nondim):
        M_out_d2nd = nondim.M["state"]["d2nd"] * nondim.nt

        # fcn_dim is already bound with params/fcns by resolve_functions
        self.fcn = nondim.nondim_function(self.fcn_dim, nondim.M["state"]["nd2d"], nondim.M["ctrl"]["nd2d"], M_out_d2nd)


# ===============================================================
# GENERALIZED LIBRARY 
# ===============================================================


# ==========================================
# -- TYPES OF OBSTACLES TO CONSIDER 
# ==========================================s
# =============================================

# -- CONSTRAINT_TEMPLATE: 

# -- AFFINE:
# ==== POLYTOPE CONSTRAINTS ========
# -- POLYTOPE_IN: 
#    --- BOX_IN:
#    --- UPPER_IN:
#    --- LOWER_IN:
#    --- ZONOTOPE_IN (not finished)
# -- POLYTOPE_OUT: 
#    --- BOX_OUT:
#    --- UPPER_OUT:
#    --- LOWER_OUT:
#    --- ZONOTOPE_OUT (not finished)
# ==== SOC Constraints ========
# -- SOC_IN: 
#    --- SPHERE_IN 
#    --- ELLIPSE_IN
# -- SOC_OUT: nonconvex approx
# -- SPHERE_IN: convex
# -- SPHERE_OUT: nonconvex approx
# -- ELLIPSE_IN: convex
# -- ELLIPSE_OUT: nonconvex approx
# -- CYLINDER_IN: convex (application of ELLIPSE_IN)
# -- CYLINDER_OUT: nonconvex approx (application of ELLIPSE_OUT)
# -- PROX_IN: convex
# -- PROX_OUT: nonconvex approx
# ------ specific CONES -------
# -- CONE_IN: convex
# -- CONE_OUT: nonconvex approx
# -- SPHERE_CONE_IN: convex
# -- SPHERE_CONE_OUT: nonconvex approx
# -- ELLIPSE_CONE_IN: convex
# -- ELLIPSE_CONE_OUT: nonconvex approx

# =========================================================
# GENERAL CONSTRAINT TEMPLATE
# =========================================================


# class CONSTRAINT_TEMPLATE:
#     def __init__(self,params={}):
#         self.type = 'specific_type'
#         self.name = 'name1' ## unique identifier
#         self.set = 'state' ; ##['state','control']
#         self.idx = [];
#         #### specific parameters options 
#         self.A = [];
#         self.C = [];
#         self.b = [];
#         self.d = [];

#         # parameters
#         self.name = name
#         self.set = set
#         self.x = x
#         self.x_idx = x_idx
#         self.boundary = boundary
#         self.idx = 0 if boundary == 'init' else -1 if boundary == 'final' else None
#         if eps is not None:
#             self.eps = eps
#         else:
#             self.eps = np.zeros(len(x_idx))
#         self.dimension = len(x_idx)

#         # options: ['affine','polytope_in','polytope_out']
#         # tag that says the exact type of obstacle 
#         self.convex = True: # options: [True,False];
#         self.category = 'affine' # options: ['affine','polytope,'soc','SDP','nonconvex']
#         self.A = params['A']
#         self.b = params['b']
#         self.C = params['C']
#         self.d = params['d']
#         # [[add other fields]]
#     ### 2) standard linearization functions - analytical solutions and jax
#     def g(self,x): pass ### - returns 0th order term for linearization
#     def dgdz(self,x): pass ### - returns 1st order term for linearization
#     def g_jax(self,x): pass ### - returns 0th order term for linearization (jax)
#     def dgdz_jax(self,x): pass ### - returns 1st order term for linearization (jax)
#     ### 3) other convexification strategies for specific case
#     def affineApprox(self,x,version='version1'): pass ### - returns alternative affine approximation other than linearization
#     def affineApprox_jax(self,x,version='version1'): pass ### - returns alternative affine approximation other than linearization (jax)
#     def cvxApprox(self,x,version='version1'): pass ### - returns alternative convex approximations other than lineariation  or self.affineApprox
#     def cvxApprox_jax(self,x,version='version1'): pass ### - returns alternative convex approximations other than lineariation  or self.affineApprox (jax)


# =========================================================
# GENERAL AFFINE CONSTRAINT
# =========================================================

class AFFINE: ## IMPLEMENTED 
    # -- defines affine constraint of the form Ax = b with A = linear and b = offset
    def __init__(self,ins={},params=None):
        # implements: Ax = b
        self.name = ins['name']; ## unique identifier
        self.b_dim = ins['b'];
        self.A_dim = ins['A'];
        self.n = self.A.shape[1];
        self.m = self.A.shape[0];
        #############################################
        self.type = 'AFFINE'
        self.implement_type = 'AFFINE'
        self.subtype = 'AFFINE'
        self.set = 'state' ; ## from ['state','control']
        self.idx = list(range(self.n)); ## index 
        self.convex = True;
        if 'type' in ins: self.type = ins['type'];
        if 'set' in ins: self.set = ins['set'];
        if 'idx' in ins: self.idx = ins['idx'];
        self.eps = 0.0001;
        if 'eps' in ins: self.eps = ins['eps'];

        #############################################
        self.M_out = np.eye(self.m);
        self.M_in = np.eye(self.n);
        if 'M_out' in ins: self.M_out = ins['M_out'];
        if 'M_in' in ins: self.M_in = ins['M_in'];
        self.M_out_inv = np.linalg.inv(self.M_out);
        #############################################
        self.A = self.M_out_inv@self.A_dim@self.M_in
        self.b = self.M_out_inv@self.b_dim;
    ### non dimensionalized version... 
    def g(self,x): return self.A@x - self.b;
    def dgdx(self,x): return self.A;
    def g_jax(self,x): return self.A@x - self.b
    def dgdx_jax(self,x): return self.A
    ################################################
    def g_aff(self,t,z,nu):
        dgdz = np.zeros([self.dimension,len(z)]);
        dgdu = np.zeros([self.dimension,len(nu)])
        if self.set == 'state': g = self.g(z[self.idx]); dgdz[:,self.idx] = self.dgdx(z[self.idx]);
        if self.set == 'control': g = self.g(nu[self.idx]); dgdu[:,self.idx] = self.dgdx(nu[self.idx]);
        return g,dgdz,dgdu
    ################################################
    def g_aff_jax(self,t,z,nu):
        dgdz = jnp.zeros([self.dimension,len(z)]);
        dgdu = jnp.zeros([self.dimension,len(nu)])
        if self.set == 'state': g = self.g_jax(z[self.idx]); dgdz[:,self.idx] = self.dgdx_jax(z[self.idx]);
        if self.set == 'control': g = self.g_jax(nu[self.idx]); dgdu[:,self.idx] = self.dgdx_jax(nu[self.idx]);
        return g,dgdz,dgdu
    ################################################

# =========================================================
# =========================================================
# =========================================================
#              GENERAL POLYTOPE CONSTRAINTS
# =========================================================
# =========================================================
# =========================================================

class POLYTOPE: ## IMPLEMENTED 
    # -- defines polytope constraint of the form Ax - b <= 0
    def __init__(self,ins={},params=None):

        self.name = ins['name'] ## unique identifier
        self.b_dim = ins['b'];
        self.A_dim = ins['A'];
        version = ins['version'];
        self.version = version; 
        self.n = self.A_dim.shape[1];
        self.m = self.A_dim.shape[0];
        self.sharp = 2;
        self.type = 'POLYTOPE';
        #############################################

        self.set = 'state' ; ## from ['state','control']
        self.idx = list(range(self.n)); ## index 
        if 'name' in ins: self.name = ins['name'];
        if 'set' in ins: self.set = ins['set'];
        if 'idx' in ins: self.idx = ins['idx'];
        if 'sharp' in ins: self.sharp = ins['sharp'];
        self.time_steps = 'all';
        if 'tsteps' in ins: self.time_steps = ins['tsteps'];
        if 'time_steps' in ins: self.time_steps = ins['time_steps'];
        ################################################
        self.M_out = np.eye(self.m);
        self.M_in = np.eye(self.n);
        if 'M_out' in ins: self.M_out = ins['M_out'];
        if 'M_in' in ins: self.M_in = ins['M_in'];
        self.M_out_inv = np.linalg.inv(self.M_out);
        #############################################
        #############################################
        self.A = self.M_out_inv@self.A_dim@self.M_in
        self.b = self.M_out_inv@self.b_dim;

        if version == 'in':
            self.implement_type = 'POLYTOPE'
            self.subtype = 'POLYTOPE_IN';
            self.convex = True;
            self.dimension = self.n;
            self.eps = 0.001;
            if 'eps' in ins: self.eps = ins['eps'];
            if not(isinstance(self.eps,(np.ndarray,jnp.ndarray))):
                self.eps = self.eps*np.ones(self.dimension)
        ################################################
        if version == 'out':
            self.implement_type = 'nonconvex_inequality'
            self.subtype = 'POLYTOPE_OUT';
            self.convex = False;
            self.dimension = 1;        
            self.eps = 0.001;
            if 'eps' in ins: self.eps = ins['eps']
            if isinstance(self.eps,(np.ndarray,jnp.ndarray)):
                temp = self.M_out_inv@self.eps; self.eps = np.min(temp);

        ## epsilon nondim logic...
        # M_out_inv @ A @ M_in @x_nodim - M_out_inv @ b  <= eps_scalar*1 <= M_out_inv @ eps 
        # where eps_scalar = min_j [M_out_inv@eps]_j
    def g(self,x):
        if self.version == 'in': return self.A@x - self.b; # g(x) # NEVER USED - CONVEX
        if self.version == 'out': # max_j g_j(x)
            z = self.A@x - self.b; 
            exps = np.exp(self.sharp*z) # softmax: implements approximation of exps = np.max(Cx - d); 
            return (1./np.sum(exps))*exps*z
    def dgdx(self,x):
        if self.version == 'in': return self.A
        if self.version == 'out': 
            z = self.A@x - self.b;
            exps = np.exp(self.sharp*z)
            summ = np.sum(exps);
            dzdx = self.A;
            dgdz = (1./summ)*exps
            dgdz = dgdz + (1./summ)*z*exps*self.sharp
            dgdz = dgdz + (z@exps)*(-1./summ**2)*self.sharp*exps
            dgdx = dgdz@dzdx;
            return dgdx
    ################################################
    def g_jax(self,x):
        if self.version == 'in': return self.A@x - self.b;
        if self.version == 'out': 
            z = self.A@x - self.b; 
            exps = jnp.exp(self.sharp*z) # softmax: implements approximation of exps = np.max(Cx - d); 
            return (1./jnp.sum(exps))*exps*z
    def dgdx_jax(self,x):
        if self.version == 'in': return self.A
        if self.version == 'out': 
            z = self.A@x - self.b;
            exps = jnp.exp(self.sharp*z)
            summ = jnp.sum(exps);
            dzdx = self.A;
            dgdz = (1./summ)*exps
            dgdz = dgdz + (1./summ)*z*exps*self.sharp
            dgdz = dgdz + (z@exps)*(-1./summ**2)*self.sharp*exps
            dgdx = dgdz@dzdx;
            return dgdx
    ################################################
    def g_aff(self,t,z,nu):
        dgdz = np.zeros([self.dimension,len(z)]);
        dgdu = np.zeros([self.dimension,len(nu)])
        if self.set == 'state': g = self.g(z[self.idx]); dgdz[:,self.idx] = self.dgdx(z[self.idx]);
        if self.set == 'control': g = self.g(nu[self.idx]); dgdu[:,self.idx] = self.dgdx(nu[self.idx]);
        return g,dgdz,dgdu
    ################################################
    def g_aff_jax(self,t,z,nu):
        dgdz = jnp.zeros([self.dimension,len(z)]);
        dgdu = jnp.zeros([self.dimension,len(nu)])
        if self.set == 'state': g = self.g_jax(z[self.idx]); dgdz[:,self.idx] = self.dgdx_jax(z[self.idx]);
        if self.set == 'control': g = self.g_jax(nu[self.idx]); dgdu[:,self.idx] = self.dgdx_jax(nu[self.idx]);
        return g,dgdz,dgdu
    ################################################    




class SOC: ## IMPLEMENTED 
    # -- defines polytope constraint of the form ||Ax + b||_2 - (Cx + d) <= 0
    # note here that C must be 1 x n
    def __init__(self,ins={},params=None):
        self.name = ins['name'] ## unique identifier
        self.A_dim = ins['A'];
        self.b_dim = ins['b'];
        self.C_dim = ins['C'];
        self.d_dim = ins['d'];
        version = ins['version'];
        self.version = version; 

        self.n = self.A_dim.shape[1];
        self.m = self.A_dim.shape[0];
        self.type = 'SOC';
        #############################################
        if version == 'in':  self.subtype = 'SOC_IN';  self.convex = True;
        if version == 'out': self.subtype = 'SOC_OUT'; self.convex = False;
        
        self.set = 'state' ; ## from ['state','control']
        self.idx = list(range(self.n)); ## index 
        if 'name' in ins: self.name = ins['name'];
        if 'set' in ins: self.set = ins['set'];
        if 'idx' in ins: self.idx = ins['idx'];
        self.eps = 0.0001;
        if 'eps' in ins: self.eps = ins['eps'];
        ################################################
        self.time_steps = 'all';
        if 'tsteps' in ins: self.time_steps = ins['tsteps'];
        if 'time_steps' in ins: self.time_steps = ins['time_steps'];

        ################################################
        self.epsilon_grad = 0.0000001;
        ################################################
        self.M_out = 1;
        self.M_in = np.eye(self.n);
        if 'M_out' in ins: self.M_out = ins['M_out'];
        if 'M_in' in ins: self.M_in = ins['M_in'];
        self.M_out_inv = 1./self.M_out;
        ## for a second order cone M_out must be a positive scalar
        #############################################
        #############################################
        self.A = self.M_out_inv*self.A_dim@self.M_in
        self.b = self.M_out_inv*self.b_dim;

        self.C = self.M_out_inv*self.C_dim@self.M_in
        self.d = self.M_out_inv*self.d_dim;


    def g(self,x): return np.linalg.norm(self.A@x + self.b) - (self.C@x + self.d);
    def dgdx(self,x): return (1./(np.linalg.norm(self.A@x + self.b)+self.epsilon_grad))*(self.A@x+self.b).T@self.A - self.C;
    def g_jax(self,x): return jnp.linalg.norm(self.A@x + self.b) - (self.C@x + self.d);
    def dgdx_jax(self,x): return (1./(jnp.linalg.norm(self.A@x + self.b)+self.epsilon_grad))*(self.A@x+self.b).T@self.A - self.C;
    ################################################
    def g_aff(self,t,z,nu):
        dgdz = np.zeros([self.dimension,len(z)]);
        dgdu = np.zeros([self.dimension,len(nu)])
        if self.set == 'state': g = self.g(z[self.idx]); dgdz[:,self.idx] = self.dgdx(z[self.idx]);
        if self.set == 'control': g = self.g(nu[self.idx]); dgdu[:,self.idx] = self.dgdx(nu[self.idx]);
        return g,dgdz,dgdu
    ################################################
    def g_aff_jax(self,t,z,nu):
        dgdz = jnp.zeros([self.dimension,len(z)]);
        dgdu = jnp.zeros([self.dimension,len(nu)])
        if self.set == 'state': g = self.g_jax(z[self.idx]); dgdz[:,self.idx] = self.dgdx_jax(z[self.idx]);
        if self.set == 'control': g = self.g_jax(nu[self.idx]); dgdu[:,self.idx] = self.dgdx_jax(nu[self.idx]);
        return g,dgdz,dgdu
    ################################################        


# =========================================================
# =========================================================
#              SPECIFIC POLYTOPE SUBTYPES
# =========================================================
# =========================================================
class BOX(POLYTOPE):
    def __init__(self,ins={},params=None):    
        upper = ins['upper'];
        lower = ins['lower'];
        # lower <= x <= upper;
        n = len(upper);
        b = np.hstack([upper,-lower]);
        A = np.vstack([np.eye(n),-np.eye(n)]);
        POLYTOPE.__init__(self,ins={'A':A,'b':b,**ins},params=params)
        self.n = n
        self.lower = lower;
        self.upper = upper;                
        if self.version == 'in': self.subtype = 'BOX_IN'
        if self.version == 'out': self.subtype = 'BOX_OUT'

class UPPER(POLYTOPE):
    def __init__(self,ins={},params=None):
        # x <= up;
        upper = ins['upper']
        n = len(upper)
        A = np.eye(n); b = upper        
        POLYTOPE.__init__(self,ins={'A':A,'b':b,**ins},params=params)
        self.n = n 
        self.upper = upper;
        if self.version == 'in': self.subtype = 'UPPER_IN'
        if self.version == 'out': self.subtype = 'UPPER_OUT'
class LOWER(POLYTOPE):
    def __init__(self,ins={},params=None):
        # low <= x
        lower = ins['lower'];
        n = len(lower)
        A = -np.eye(n); b = -lower
        POLYTOPE.__init__(self,ins={'A':A,'b':b,**ins},params=params)
        self.n = n
        self.lower = lower        
        if self.version == 'in': self.subtype = 'LOWER_IN'
        if self.version == 'out': self.subtype = 'LOWER_OUT'

# =========================================================
# ----------- SPECIFIC POLYTOPES TO BE ADDED  
# ===========================s==============================
class ZONOTOPE:  ### NOT FINISHED 
    ## defines convex zonotope constraint
    ## zonotope: a type of polytope constraint with specific structure
    def __init__(self): pass 
# =============================================================
# =============================================================
# =============================================================
#         GENERAL SECOND ORDER CONE (SOC) CONSTRAINTS
# =============================================================
# =============================================================
# =============================================================

# =========================================================
# =========================================================
# ----------- SPECIFIC SOC CONSTRAINTS
# =========================================================
# =========================================================
class SPHERE(SOC): 
    # -- Description: defines a spherical "keep in" region
    ## General SOC form: ||Ax+b||_2 <= Cx + d;
    def __init__(self,ins={},params=None):
        center = ins['center']
        radius = ins['radius']
        ################################
        n = len(center)
        A = np.eye(n); b = -center;
        C = np.zeros(n); d = radius;
        SOC.__init__(self,ins={'A':A,'b':b,'C':C,'d':d,**ins},params=params);
        self.center = center;
        self.radius = radius;
        self.n = n
        if self.version == 'in': self.subtype = 'SPHERE_IN'
        if self.version == 'out': self.subtype = 'SPHERE_OUT'

class ELLIPSOID(SOC):
    ## -- defines a ellipsoidal keep in region
    def __init__(self,ins={},params=None): 
        #### REQUIRED ####
        center = ins['center']
        U = ins['U']
        ########################
        diag = []
        if 'diag' in ins: diag = ins['diag'];
        if len(diag)==0: diag = np.ones(len(U));
        n = len(center)
        D = np.diag(diag);
        Dinv = np.diag(1./diag);
        A = Dinv@U;
        b = -Dinv@U@center
        C = np.zeros(n);
        d = 1.;
        SOC.__init__(self,ins={'A':A,'b':b,'C':C,'d':d,**ins},params=params);
        self.center = center;
        self.U = U;
        self.diag = diag;
        self.D = D
        self.Dinv = Dinv;
        self.n = n;
        if self.version == 'in': self.subtype = 'ELLIPSOID_IN'
        if self.version == 'out': self.subtype = 'ELLIPSOID_OUT'

class CYLINDER(ELLIPSOID,SOC):
    ## -- defines a ellipsoidal keep in region
    def __init__(self,ins={},params=None): 
        #### REQUIRED #### 
        center = ins['center'];
        U = ins['U'];
        ### OPTIONAL ####
        radius = None; diag = [];
        if 'radius' in ins: radius = ins['radius'];
        if 'diag' in ins: diag = ins['diag'];
        ############################
        if radius == None: radius = 1;
        temp = radius*np.ones(len(U));
        if len(diag)==0: diag = temp;

        ELLIPSOID.__init__(self,ins={'center':center,'U':U,'diag':diag,**ins},params=params);
        self.radius = radius;
        if self.version == 'in': self.subtype = 'CYLINDER_IN';
        if self.version == 'out': self.subtype = 'CYLINDER_OUT';

class CONE(SOC):
    def __init__(self,ins={},params=None):
        #### REQUIRED ##### 
        center = ins['center']
        U = ins['U']
        N = ins['N']
        theta = ins['theta']
        ### OPTIONAL ####
        radius = None; diag = [];
        if 'radius' in ins: radius = ins['radius'];
        if 'diag' in ins: diag = ins['diag'];
        ##########################

        if radius == None: radius = 1;
        temp = radius*np.ones(len(U));
        if len(diag)==0: diag = temp;
        n = len(U);
        D = np.diag(diag);
        Dinv = np.diag(1./diag);
        A = Dinv@U;
        b = -Dinv@U@center
        C = np.tan(theta)*N
        d = -np.tan(theta)*N@center;
        SOC.__init__(self,ins={'A':A,'b':b,'C':C,'d':d,**ins},params=params);
        self.center = center;
        self.U = U;
        self.diag = diag;
        self.D = D
        self.Dinv = Dinv;
        self.n = n;
        self.theta = theta;
        self.N = N;
        self.radius = radius;
        if self.version == 'in': self.subtype = 'CONE_IN'
        if self.version == 'out': self.subtype = 'CONE_OUT'

class PROXIMITY(SOC):
    ## -- defines a ellipsoidal keep in region
    def __init__(self,ins={},params=None):
        U=[]; radius=None; idx1=[]; diag=[];
        if 'U' in ins: U = ins['U']
        if 'radius' in ins: radius = ins['radius'];
        if 'idx1' in ins: idx1 = ins['idx1'];
        if 'diag' in ins: diag = ins['diag'];

        if len(U) == 0: nx = len(idx1); U = np.eye(nx);
        if radius == None: radius = 1;
        if len(diag)==0: diag = radius*np.ones(len(U));
        nx = len(U[0]); EYE = np.eye(nx);
        if len(idx1)==0: idx1 = list(range(int(nx)));
        totinds = list(range(int(nx*2)));
        idx2 = totinds.copy();
        [idx2.remove(ind) for ind in idx1];
        DIFF = np.zeros([nx,nx2]);
        DIFF[:,idx1] = EYE; DIFF[:,idx2] = EYE;
        n = len(U);
        D = np.diag(diag);
        Dinv = np.diag(1./diag);
        A = Dinv@U@DIFF;
        b = np.zeros(len(U));
        C = np.zeros(n);
        d = 1.;
        SOC.__init__(self,ins={'A':A,'b':b,'C':C,'d':d,**ins},params=params);
        self.center = center;
        self.U = U;
        self.diag = diag;
        self.D = D
        self.Dinv = Dinv;
        self.n = n;
        if self.version == 'in': self.subtype = 'PROXIMITY_IN'
        if self.version == 'out': self.subtype = 'PROXIMITY_OUT'

################################################################################
##### ------------------------------------------------------------------- ######
##### ------------------------------------------------------------------- ######
##### ------------------------------------------------------------------- ######
##### ------------------------------------------------------------------- ######
##### ------------------------------------------------------------------- ######
##### ------------------------------------------------------------------- ######
##### ------------------------------------------------------------------- ######
##### ------------------------------------------------------------------- ######
##### ------------------------------------------------------------------- ######
##### ------------------------------------------------------------------- ######
##### ------------------------------------------------------------------- ######
##### ------------------------------------------------------------------- ######
################################################################################


# =========================================================
# ----------- CONE CONSTRAINTS 
# =========================================================

# ------ specific CONES -------
# -- CONE_IN: convex
# -- CONE_OUT: nonconvex approx
# -- SPHERE_CONE_IN: convex
# -- SPHERE_CONE_OUT: nonconvex approx
# -- ELLIPSE_CONE_IN: convex
# -- ELLIPSE_CONE_OUT: nonconvex approx




### DAN'S REWRITE OF CARLOS'S CONSTRAINTS TO SYNC UP... 
# ===============================================================
# CONVEX CONSTRAINTS
# ===============================================================

class dan_equality_bc(AFFINE):
    def __init__(self, name, set, x, x_idx, boundary, eps=None, params=None):
        # parameters
        self.name = name
        self.set = set
        self.x_dim = x
        self.x_idx = x_idx
        self.boundary = boundary
        self.idx = 0 if boundary == 'init' else -1 if boundary == 'final' else None
        if eps is not None: self.eps = eps
        else: self.eps = np.zeros(len(x_idx))
        self.dimension = len(x_idx)
        self.x = None
        ### NEW 
        A = np.eye(len(x))
        b = self.x;
        AFFINE.__init__(self,name,A,b,version=version,ins=ins)
    # written for nondim input
    def fcn(self, x):
        return x[self.x_idx] - self.x
    def nondim_constraint(self, nondim):
        if self.set == "state":
            self.x = nondim.M["state"]["d2nd"][np.ix_(self.x_idx, self.x_idx)] @ self.x_dim
        elif self.set == "control":
            self.x = nondim.M["ctrl"]["d2nd"][np.ix_(self.x_idx, self.x_idx)] @ self.x_dim



# class AFFINE: 
#     # -- defines affine constraint of the form Ax = b with A = linear and b = offset
#     def __init__(self,name,A,b,ins={}):
#         # implements: Ax = b
#         self.name = name; ## unique identifier
#         self.b = b;
#         self.A = A;
#         self.n = self.A.shape[1];
#         self.m = self.A.shape[0];
#         #############################################
#         self.type = 'AFFINE'
#         self.subtype = 'AFFINE'
#         self.set = 'state' ; ## from ['state','control']
#         self.idx = list(range(self.n)); ## index 
#         self.convex = True;
#         if 'type' in ins: self.type = ins['type'];
#         if 'set' in ins: self.set = ins['set'];
#         if 'idx' in ins: self.idx = ins['idx'];
#         #############################################
#     ### standard linearization
#     def g(self,x): return self.A@x - self.b;
#     def dgdx(self,x): return self.A
#     def g_jax(self,x): return self.A@x - self.b
#     def dgdx_jax(self,x): return self.A
        
#         self.n = n
#         self.lower = lower;
#         self.upper = upper;                
#         if version == 'in': self.subtype = 'BOX_IN'
#         if version == 'out': self.subtype = 'BOX_OUT'



class dan_inequality_bc(POLYTOPE):
    def __init__(self, name, set, x_min, x_min_idx, x_max, x_max_idx, boundary, eps=np.array([]), params=None):

        # parameters
        self.name = name
        self.set = set
        self.x_min_dim = x_min
        self.x_min_idx = x_min_idx
        self.x_max_dim = x_max
        self.x_max_idx = x_max_idx
        self.idx = 0 if boundary == 'init' else -1 if boundary == 'final' else None
        self.eps = eps
        self.dimension = len(x_min_idx) + len(x_max_idx)
        self.x_min = None
        self.x_max = None

    def nondim_constraint(self, nondim):
        if self.set == "state":
            self.x_min = nondim.M["state"]["d2nd"][np.ix_(self.x_min_idx, self.x_min_idx)] @ self.x_min_dim
            self.x_max = nondim.M["state"]["d2nd"][np.ix_(self.x_max_idx, self.x_max_idx)] @ self.x_max_dim
        elif self.set == "control":
            self.x_min = nondim.M["ctrl"]["d2nd"][np.ix_(self.x_min_idx, self.x_min_idx)] @ self.x_min_dim
            self.x_max = nondim.M["ctrl"]["d2nd"][np.ix_(self.x_max_idx, self.x_max_idx)] @ self.x_max_dim

# ---------------------------------------------------------------
# path inequality constraints
# ---------------------------------------------------------------
class dan_box(BOX):
    def __init__(self, name, set, x_min, x_min_idx, x_max, x_max_idx, params=None):

        self.name = name
        self.set = set
        self.x_min_dim = x_min
        self.x_min_idx = x_min_idx
        self.x_max_dim = x_max
        self.x_max_idx = x_max_idx
        self.dimension = len(x_min_idx) + len(x_max_idx)

        self.x_min = None
        self.x_max = None

        if self.set == "state":
            n_elem = params['model']['dimensions']['n']
        elif self.set == "control":
            n_elem = params['model']['dimensions']['m']

        M_min = -np.eye(n_elem)[self.x_min_idx, :]
        M_max = np.eye(n_elem)[self.x_max_idx, :]

        self.M_select = np.vstack([M_min, M_max])

    def nondim_constraint(self, nondim):
        if self.set == "state":
            self.x_min = nondim.M["state"]["d2nd"][np.ix_(self.x_min_idx, self.x_min_idx)] @ self.x_min_dim
            self.x_max = nondim.M["state"]["d2nd"][np.ix_(self.x_max_idx, self.x_max_idx)] @ self.x_max_dim

        if self.set == "control":
            self.x_min = nondim.M["ctrl"]["d2nd"][np.ix_(self.x_min_idx, self.x_min_idx)] @ self.x_min_dim
            self.x_max = nondim.M["ctrl"]["d2nd"][np.ix_(self.x_max_idx, self.x_max_idx)] @ self.x_max_dim

# ---------------------------------------------------------------
# rate constraints
# ---------------------------------------------------------------
class dan_control_rate_limit(UPPER):
    def __init__(self, name, udot_max, udot_max_idx, params=None):
        self.name = name
        self.udot_max_dim = udot_max
        self.udot_max_idx = udot_max_idx
        self.dimension = len(udot_max_idx)

        n_elem = params['model']['dimensions']['m']
        M_min = -np.eye(n_elem)[self.udot_max_idx, :]
        M_max = np.eye(n_elem)[self.udot_max_idx, :]

        self.udot_max = None

        self.M_select = np.vstack([M_min, M_max])

    def nondim_constraint(self, nondim):
        self.udot_max = nondim.nt * nondim.M["ctrl"]["d2nd"][np.ix_(self.udot_max_idx, self.udot_max_idx)] @ self.udot_max_dim

# ---------------------------------------------------------------
# Second-order cone cosntraints
# ---------------------------------------------------------------

class dan_axis_angle_cone(CONE,SOC):
    def __init__(self, name, set, axis, theta_max, x_idx, params=None):
        self.name = name
        self.set = set
        self.axis = axis / np.linalg.norm(axis)
        self.cos_theta_max = np.cos(np.deg2rad(theta_max))
        self.x_idx = x_idx
        self.dimension = 1

    def nondim_constraint(self, nondim):
        # the deg2rad is already nondimming
        pass

class dan_max_norm_cone(CONE,SOC):
    def __init__(self, name, set, max_val, x_idx, params=None):
        self.name = name
        self.set = set
        self.max_val_dim = max_val
        self.x_idx = x_idx
        self.dimension = 1

        self.max_val = None

    def nondim_constraint(self, nondim):

        if self.set == "state":
            nondim_key = nondim.z_types[self.x_idx[0]]
            self.max_val = self.max_val_dim * nondim.scales[nondim_key]
        
        elif self.set == "control":
            nondim_key = nondim.u_types[self.x_idx[0]]
            self.max_val =  self.max_val_dim * nondim.scales[nondim_key]

class dan_quaternion_cone(CONE,SOC):
    def __init__(self, name, theta_max, axis_num, quat_start_idx, params=None):
        self.name = name
        self.quat_start_idx = quat_start_idx
        self.cos_theta_max = np.cos(np.deg2rad(theta_max))
        self.axis_num = axis_num
        self.rhs = np.sqrt((1.0 - self.cos_theta_max) * 0.5)
        self.dimension = 1

    def nondim_constraint(self, nondim):
        pass

# ===============================================================
# NONCONVEX CONSTRAINTS
# ===============================================================

# TODO: change to (func - max_val) / scale
class dan_custom_nonconvex_inequality(POLYTOPE):
    def __init__(self, name, group, fcn, units, dimension, ct, eps=None, max_val=None, mission_params=None, params=None):
        self.name = name
        self.group = group
        self.mission_params = mission_params
        self.params = params
        self.units = units
        self.eps = eps
        self.dimension = dimension
        self.ct = ct

        self.fcn_dim = resolve_function(fcn)
        
        # this will be a function f(t, z, nu)
        self.fcn = None
        
        self.fcn_jit = None
        self.dfcn_dz_jit = None
        self.dfcn_du_jit = None

        self.has_max_val = max_val is not None
        
        if self.has_max_val is False:
            self.max_val_dim = None
        else:
            self.max_val_dim = jnp.atleast_1d(jnp.asarray(max_val))
    
    def g_aff(self, t, z, nu):
        return self.fcn_jit(z, nu), self.dfcn_dz_jit(z, nu), self.dfcn_du_jit(z, nu)

    def nondim_constraint(self, nondim):

        if self.max_val_dim is not None:
            M_out_d2nd = np.diag(1 / self.max_val_dim)
        else:
            M_out_d2nd, M_out_nd2d = nondim.build_nondim_matrix(self.units)

        if self.has_max_val:
            self.max_val = M_out_d2nd @ self.max_val_dim

        # fcn_dim is already bound with params/fcns by resolve_functions
        nd_fcn_lhs = nondim.nondim_function(self.fcn_dim, nondim.M["state"]["nd2d"], nondim.M["ctrl"]["nd2d"], M_out_d2nd)

        if self.has_max_val:
            self.fcn = lambda t, z, nu: nd_fcn_lhs(t, z, nu) - self.max_val
        else:
            self.fcn = lambda t, z, nu: nd_fcn_lhs(t, z, nu)




