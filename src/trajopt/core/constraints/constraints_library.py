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

        self.x = None

    # written for nondim input
    def fcn(self, x):
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

class AFFINE: 
    # -- defines affine constraint of the form Ax = b with A = linear and b = offset
    def __init__(self,b,A,ins={}):
        # implements: Ax = b
        self.b = b;
        self.A = A;
        self.n = self.A.shape[1];
        self.m = self.A.shape[0];
        #############################################
        self.type = 'AFFINE'
        self.subtype = 'AFFINE'
        self.name = 'name1' ## unique identifier
        self.set = 'state' ; ## from ['state','control']
        self.idx = list(range(self.n)); ## index 
        self.convex = True;
        if 'type' in ins: self.type = ins['type'];
        if 'name' in ins: self.name = ins['name'];
        if 'set' in ins: self.set = ins['set'];
        if 'idx' in ins: self.idx = ins['idx'];
        #############################################
    ### standard linearization
    def g(self,x): return self.A@x - self.b;
    def dgdx(self,x): return self.A
    def g_jax(self,x): pass # [[TO ADD]]
    def dgdx_jax(self,x): pass # [[TO ADD]]
    ### convex strategy standard linearization
    def affineApprox(self,x,version='standard'): return self.A,self.b
    def affineApprox_jax(self,x,version='standard'): pass # [[TO ADD]]
    def cvxApprox(self,x,version='standard'): return self.A,self.b
    def cvxApprox_jax(self,x,version='standard'): pass # [[TO ADD]]

# =========================================================
# =========================================================
# =========================================================
#              GENERAL POLYTOPE CONSTRAINTS
# =========================================================
# =========================================================
# =========================================================

class POLYTOPE:
    # -- defines polytope constraint of the form Ax - b <= 0
    def __init__(self,A,b,version='in',ins={}):
        self.b = b;
        self.A = A;
        self.n = self.A.shape[1];
        self.m = self.A.shape[0];
        self.alpha = 10;
        self.version = version;
        #############################################
        if version == 'in':  self.type = 'POLYTOPE_IN';  self.subtype = 'POLYTOPE_IN';  self.convex = True;
        if version == 'out': self.type = 'POLYTOPE_OUT'; self.subtype = 'POLYTOPE_OUT'; self.convex = False;

        self.name = 'name1' ## unique identifier
        self.set = 'state' ; ## from ['state','control']
        self.idx = list(range(self.n)); ## index 
        if 'name' in ins: self.name = ins['name'];
        if 'set' in ins: self.set = ins['set'];
        if 'idx' in ins: self.idx = ins['idx'];
        if 'alpha' in ins: self.alpha = ins['alpha'];
        #############################################
    def g(self,x):
        if self.version == 'in': return self.A@x - self.b;
        if self.version == 'out': 
            z = self.A@x - self.b; 
            exps = np.exp(self.alpha*z) # softmax: implements approximation of exps = np.max(Cx - d); 
            return (1./np.sum(exps))*exps*z
    def dgdx(self,x):
        if self.version == 'in': return self.A
        if self.version == 'out': 
            z = self.A@x - self.b;
            exps = np.exp(self.alpha*z)
            summ = np.sum(exps);
            dzdx = self.A;
            dgdz = (1./summ)*exps
            dgdz = dgdz + (1./summ)*z*exps*self.alpha
            dgdz = dgdz + (z@exps)*(-1./summ**2)*self.alpha*exps
            dgdx = dgdz@dzdx;
            return dgdx

    def g_jax(self,x): pass # [[TO ADD]]
    def dgdx_jax(self,x): pass # [[TO ADD]]
    ### convex strategy standard linearization
    def affineApprox(self,x,version='standard'): return self.b,self.A
    def affineApprox_jax(self,x,version='standard'): pass # [[TO ADD]]
    def cvxApprox(self,x,version='standard'): return self.d,self.C
    def cvxApprox_jax(self,x,version='standard'): pass # [[TO ADD]]

# =========================================================
# =========================================================
#              SPECIFIC POLYTOPE SUBTYPES
# =========================================================
# =========================================================
class BOX(POLYTOPE):
    def __init__(self,low,up,version='in',ins={}):
        # low <= x <= up;
        n = len(up);
        b = np.hstack([up,-low]);
        A = np.vstack([np.eye(n),-np.eye(n)]);
        POLYTOPE.__init__(self,A,b,version=version,ins=ins)
        self.n = n
        self.low = low;
        self.up = up;                
        if version == 'in': self.subtype = 'BOX_IN'
        if version == 'out': self.subtype = 'BOX_OUT'
class UPPER(POLYTOPE):
    def __init__(self,up,version='in',ins={}):
        # x <= up;
        n = len(up)
        A = np.eye(n); b = up        
        POLYTOPE.__init__(self,A,b,version=version,ins=ins)
        self.n = n 
        self.up = up;
        if version == 'in': self.subtype = 'UPPER_IN'
        if version == 'out': self.subtype = 'UPPER_OUT'
class LOWER(POLYTOPE):
    def __init__(self,low,version='in',ins={}):
        # low <= x
        n = len(low)
        A = -np.eye(n); b = -low
        POLYTOPE.__init__(self,A,b,version=version,ins=ins)
        self.n = n
        self.low = low        
        if version == 'in': self.subtype = 'LOWER_IN'
        if version == 'out': self.subtype = 'LOWER_OUT'

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

class SOC:
    def __init__(self,A,b,C,d,version='in',ins={}):
        self.A = A;
        self.b = b;
        self.C = C;
        self.d = d;
        self.n = self.A.shape[1];
        self.m = self.A.shape[0];
        # self.alpha = 10;
        self.version = version;
        #############################################
        if version == 'in':  self.type = 'SOC_IN';  self.subtype = 'SOC_IN';  self.convex = True;
        if version == 'out': self.type = 'SOC_OUT'; self.subtype = 'SOC_OUT'; self.convex = False;
        self.name = 'name1' ## unique identifier
        self.set = 'state' ; ## from ['state','control']
        self.idx = list(range(self.n)); ## index 
        if 'name' in ins: self.name = ins['name'];
        if 'set' in ins: self.set = ins['set'];
        if 'idx' in ins: self.idx = ins['idx'];
        # if 'alpha' in ins: self.alpha = ins['alpha'];
        #############################################
        self.epsilon_grad = 0.0000001;
    def g(self,x): return np.linalg.norm(self.A@x + self.b) - (self.C@x + self.d);
    def dgdx(self,x): return (1./(np.linalg.norm(self.A@x + self.b)+self.epsilon_grad))*(self.A@x+self.b).T@self.A - self.C;
    def g_jax(self,x): pass # [[TO ADD]]
    def dgdx_jax(self,x): pass # [[TO ADD]]
# =========================================================
# =========================================================# ----------- SPECIFIC SOC CONSTRAINTS
# =========================================================
# =========================================================
class SPHERE(SOC): 
    # -- Description: defines a spherical "keep in" region
    ## General SOC form: ||Ax+b||_2 <= Cx + d;
    def __init__(self,center,radius,version='in',ins={}):
        n = len(center)
        A = np.eye(n); b = -center;
        C = np.zeros(n); d = radius;
        SOC.__init__(self,A,b,C,d,version=version,ins=ins);
        self.center = center;
        self.radius = radius;
        self.n = n
        self.version = version;
        if version == 'in': self.subtype = 'SPHERE_IN'
        if version == 'out': self.subtype = 'SPHERE_OUT'

class ELLIPSOID(SOC):
    ## -- defines a ellipsoidal keep in region
    def __init__(self,center,U,diag=[],version='in',ins={}):
        if len(diag)==0: diag = np.ones(len(U));
        n = len(U);
        D = np.diag(diag);
        Dinv = np.diag(1./diag);
        A = Dinv@U;
        b = -Dinv@U@center
        C = np.zeros(n);
        d = 1.;
        SOC.__init__(self,A,b,C,d,version=version,ins=ins);
        self.center = center;
        self.U = U;
        self.diag = diag;
        self.D = D
        self.Dinv = Dinv;
        self.n = n;
        self.version = version;
        if version == 'in': self.subtype = 'ELLIPSOID_IN'
        if version == 'out': self.subtype = 'ELLIPSOID_OUT'

class CYLINDER(ELLIPSOID,SOC):
    ## -- defines a ellipsoidal keep in region
    def __init__(self,center,U,radius=None,diag=[],version='in',ins={}):
        if radius == None: radius = 1;
        temp = radius*np.ones(len(U));
        if len(diag)==0: diag = temp;
        ELLIPSOID.__init__(self,center,U,diag,version=version,ins=ins);
        self.radius = radius;
        self.version = version;
        if version == 'in': self.subtype = 'CYLINDER_IN';
        if version == 'out': self.subtype = 'CYLINDER_OUT';

class CONE(SOC):
    def __init__(self,center,U,N,theta,radius=None,diag=[],version='in',ins={}):
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
        SOC.__init__(self,A,b,C,d,version=version,ins=ins);
        self.center = center;
        self.U = U;
        self.diag = diag;
        self.D = D
        self.Dinv = Dinv;
        self.n = n;
        self.theta = theta;
        self.N = N;
        self.radius = radius;
        self.version = version;
        if version == 'in': self.subtype = 'CONE_IN'
        if version == 'out': self.subtype = 'CONE_OUT'

class PROXIMITY(SOC):
    ## -- defines a ellipsoidal keep in region
    def __init__(self,U=[],radius=None,idx1=[],diag=[],version='in',ins={}):

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
        SOC.__init__(self,A,b,C,d,version=version,ins=ins);
        self.center = center;
        self.U = U;
        self.diag = diag;
        self.D = D
        self.Dinv = Dinv;
        self.n = n;
        self.version = version;
        if version == 'in': self.subtype = 'PROXIMITY_IN'
        if version == 'out': self.subtype = 'PROXIMITY_OUT'

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