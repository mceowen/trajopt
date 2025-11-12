import numpy as np
import scipy as sp
import numpy.linalg as mat
import scipy.linalg as smat

# ==========================================
# -- TYPES OF OBSTACLES TO CONSIDER 
# ==========================================s
# =============================================

# -- CONSTRAINT_TEMPLATE: 

# -- AFFINE:
# ==== POLYTOPE Constraints ========
# -- POLYTOPE_IN: convex
# -- POLYTOPE_OUT: nonconvex approx
# -- ZONOTOPE_IN: convex
# -- ZONOTOPE_OUT: nonconvex approx
# -- BOX_IN: convex
# -- BOX_OUT: nonconvex approx

# ==== SOC Constraints ========
# -- SOC_IN: convex
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


class CONSTRAINT_TEMPLATE:
    def __init__(self,params={}):
        self.type = 'specific_type'  
        # options: ['affine','polytope_in','polytope_out']
        # tag that says the exact type of obstacle 
        self.convex = True: # options: [True,False];
        self.category = 'affine' # options: ['affine','polytope,'soc','SDP','nonconvex']
        self.A = params['A']
        self.b = params['b']
        self.C = params['C']
        self.d = params['d']
        # [[add other fields]]
    ### 1) returns convex parameters required to implement convex constraints -- only for convex constraint types
    def cvxForm(self): pass
        ## --- returns parameters needed for convex constraint based on self.category
        ## - affine: return A,b for constraint Ax - b = 0
        ## - polytope: returns C,d for constraint Cx - d >= 0
        ## - soc: returns A,b,C,d  for constraint ||Cx + d || <= Ax + b
        ## - SDP returns [[to be added]]
    ### 2) standard linearization functions - analytical solutions and jax
    def g(self,x): pass 
        ### - returns 0th order term for linearization
    def dgdz(self,x): pass
        ### - returns 1st order term for linearization
    def g_jax(self,x): pass
        ### - returns 0th order term for linearization (jax)
    def dgdz_jax(self,x): pass
        ### - returns 1st order term for linearization (jax)
    ### 3) other convexification strategies for specific case
    def affineApprox(self,x,version='version1'): pass
        ### - returns alternative affine approximation other than linearization
    def affineApprox_jax(self,x,version='version1'): pass
        ### - returns alternative affine approximation other than linearization (jax)
    def cvxApprox(self,x,version='version1'): pass
        ### - returns alternative convex approximations other than lineariation  or self.affineApprox
    def cvxApprox_jax(self,x,version='version1'): pass
        ### - returns alternative convex approximations other than lineariation  or self.affineApprox (jax)

# =========================================================
# GENERAL AFFINE CONSTRAINT
# =========================================================

class AFFINE: 
    # -- defines affine constraint of the form Ax = b with A = linear and b = offset
    def __init__(self,b,A):
        self.type = 'affine'
        self.convex = True;
        self.category = 'affine'
        self.b = b
        self.A = A
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
# GENERAL POLYTOPE CONSTRAINTS
# =========================================================

class POLYTOPE_IN:
    # -- defines polytope constraint of the form Cx - d >= 0
    def __init__(self,offset,linear):
        self.type = 'polytope_in'
        self.convex = True; 
        self.category = 'polytope'  # a convex type
        self.d = d
        self.C = C;
    def g(self,x): return self.C@x - self.d;
    def dgdx(self,x): return self.C
    def g_jax(self,x): pass # [[TO ADD]]
    def dgdx_jax(self,x): pass # [[TO ADD]]
    ### convex strategy standard linearization
    def affineApprox(self,x,version='standard'): return self.d,self.C
    def affineApprox_jax(self,x,version='standard'): pass # [[TO ADD]]
    def cvxApprox(self,x,version='standard'): return self.d,self.C
    def cvxApprox_jax(self,x,version='standard'): pass # [[TO ADD]]


class POLYTOPE_OUT: 
    # -- defines polytope keep out region with polytope Cx - d >= 0
    # uses a softmax approximation
    def __init__(self,C,d):
        self.type = 'polytope_out'
        self.convex = False;
        self.category = 'nonconvex'
        self.d = d
        self.C = C
        self.alpha = 10;
    def g(self,x):
        # implements keep out region g(x) = max(Cx - d) <= 0
        # with the soft max version max_j z_j = (1/\sum_j e^{alpha z_j})*(\sum_j z_j e^{alpha z_j})
        z = self.C@x - self.d; 
        exps = np.exp(self.alpha*z) # softmax: implements approximation of exps = np.max(Cx - d); 
        out = (1./np.sum(exps))*exps*z
        return out;
    def dgdx(self,x):
        z = self.C@x - self.d;
        exps = np.exp(self.alpha*z)
        summ = np.sum(exps);
        dzdx = self.C;
        dgdz = (1./summ)*exps
        dgdz = dgdz + (1./summ)*z*exps*self.alpha
        dgdz = dgdz + (z@exps)*(-1./summ**2)*alpha*exps
        dgdx = dgdz@dzdx;
        return dgdx
    def g_jax(self,x): pass # [[TO ADD]]
    def dgdx_jax(self,x): pass # [[TO ADD]]
    def affineApprox(self,x,version='standard'): pass # [[TO ADD]]
    def affineApprox_jax(self,x,version='standard'): pass # [[TO ADD]]
    def cvxApprox(self,x,verion='standard'): pass # [[TO ADD]]
    def cvxApprox_jax(self,x,version='standard'): pass # [[TO ADD]]

# =========================================================
# ----------- SPECIFIC POLYTOPE CONSTRAINTS
# =========================================================

class BOX_IN:
    def __init__(self,dl,dh):
        # -- defines a box:  dl <= x <= dh;
        # -- rewrite: x - dl >= 0; -x + dh >= 0;
        # -- rewrite: [I; -I]x - [dl; -dh] >= 0;
        self.type = 'box_in'
        self.convex = True; 
        self.category = 'box'; # a convex type
        self.dl = dl;
        self.dh = dh;
        self.C = np.hstack([np.eye(len(dl)),-np.eye(len(dh))]);
        self.d = np.hstack([dl,-dh]);

    def g(self,x): return self.C@x - self.d;
    def dgdx(self,x): return self.C
    def g_jax(self,x): pass # [[TO ADD]]
    def dgdx_jax(self,x): pass # [[TO ADD]]
    def affineApprox(self,x,version='standard'): return self.d,self.C
    def affineApprox_jax(self,x,version='standard'): pass # [[TO ADD]]
    def cvxApprox(self,x,version='standard'): return self.d,self.C
    def cvxApprox_jax(self,x,version='standard'): pass # [[TO ADD]]

class BOX_OUT: 
    def __init__(self,dl,dh):
        self.type = 'box_out'
        self.convex = False; 
        self.category = 'nonconvex'
        self.dl = dl;
        self.dh = dh;
        self.C = np.hstack([np.eye(len(dl)),-np.eye(len(dh))]);
        self.d = np.hstack([dl,-dh]);
        self.alpha = 10;

    def g(self,x):
        # implements keep out region g(x) = max(Cx - d) <= 0
        # with the soft max version max_j z_j = (1/\sum_j e^{alpha z_j})*(\sum_j z_j e^{alpha z_j})
        z = self.C@x - self.d; 
        exps = np.exp(self.alpha*z) # softmax: implements approximation of exps = np.max(Cx - d); 
        out = (1./np.sum(exps))*exps*z
        return out;
    def dgdx(self,x):
        z = self.C@x - self.d;
        exps = np.exp(self.alpha*z)
        summ = np.sum(exps);
        dzdx = C;
        dgdz = (1./summ)*exps
        dgdz = dgdz + (1./summ)*z*exps*self.alpha
        dgdz = dgdz + (z@exps)*(-1./summ**2)*alpha*exps
        dgdx = dgdz@dzdx;
        return dgdx
    def g_jax(self,x): pass # [[TO ADD]]
    def dgdx_jax(self,x): pass # [[TO ADD]]
    def affineApprox(self,x,version='standard'): pass # [[TO ADD]]
    def affineApprox_jax(self,x,version='standard'): pass # [[TO ADD]]
    def cvxApprox(self,x,verion='standard'): pass # [[TO ADD]]
    def cvxApprox_jax(self,x,version='standard'): pass # [[TO ADD]]


# =========================================================
# ----------- SPECIFIC POLYTOPES TO BE ADDED  
# =========================================================
class ZONOTOPE_IN:  ### NOT FINISHED 
    ## defines convex zonotope constraint
    ## zonotope: a type of polytope constraint with specific structure
    def __init__(self):
        self.type = 'zonotope_in'
        self.convex = True; 
        self.category = 'zonotope'
        ## add more 
    def g(self,x): pass # [[TO ADD ]]
    def dgdx(self,x): pass # [[TO ADD]]
    def g_jax(self,x): pass # [[TO ADD]]
    def dgdx_jax(self,x): pass # [[TO ADD]]
    def affineApprox(self,x,version='standard'): pass 
    def affineApprox_jax(self,x,version='standard'): pass # [[TO ADD]]
    def cvxApprox(self,x,version='standard'): pass
    def cvxApprox_jax(self,x,version='standard'): pass # [[TO ADD]]

class ZONOTOPE_OUT:  ### NOT FINISHED 
    # -- defines zonotope keep out region
    def __init__(self):
        self.type = 'zonotope_out'
        self.convex = False; 
        self.category = 'nonconvex'
        ## add more 
    def g(self,x): pass # [[TO ADD ]]
    def dgdx(self,x): pass # [[TO ADD]]
    def g_jax(self,x): pass # [[TO ADD]]
    def dgdx_jax(self,x): pass # [[TO ADD]]
    def affineApprox(self,x,version='standard'): pass 
    def affineApprox_jax(self,x,version='standard'): pass # [[TO ADD]]
    def cvxApprox(self,x,version='standard'): pass
    def cvxApprox_jax(self,x,version='standard'): pass # [[TO ADD]]

# =============================================================
# GENERAL Second Order Cone (SOC) constraints
# =============================================================

class SOC_IN:
    def __init__(self,A,b,C,d):
        self.type = 'soc_in'
        self.convex = True;
        self.category = 'soc';
        self.A = A; 
        self.b = b;
        self.C = C;
        self.d = d;
    def g(self,x): return mat.norm(self.C@x + self.d) - self.A@x - self.b;
    def dgdx(self,x): pass # [[TO ADD]]
    def g_jax(self,x): pass # [[TO ADD]]
    def dgdx_jax(self,x): pass # [[TO ADD]]
    def affineApprox(self,x,version='standard'): pass # [[TO ADD]]
    def affineApprox_jax(self,x,version='standard'): pass # [[TO ADD]]
    def cvxApprox(self,x,version='standard'): return self.A,self.b,self.C.self.d
    def cvxApprox_jax(self,x,version='standard'): pass # [[TO ADD]]

class SOC_OUT:     
    def __init__(self,A,b,C,d):
        self.type = 'soc_out'
        self.convex = False;
        self.category = 'nonconvex';
        self.A = A; 
        self.b = b;
        self.C = C;
        self.d = d;
    def g(self,x): return mat.norm(self.C@x + self.d) - self.A@x - self.b;
    def dgdx(self,x): pass # [[TO ADD]]
    def g_jax(self,x): pass # [[TO ADD]]
    def dgdx_jax(self,x): pass # [[TO ADD]]
    def affineApprox(self,x,version='standard'): pass # [[TO ADD]]
    def affineApprox_jax(self,x,version='standard'): pass # [[TO ADD]]
    def cvxApprox(self,x,version='standard'): pass # [[TO ADD]]
    def cvxApprox_jax(self,x,version='standard'): pass # [[TO ADD]]        

# =========================================================
# ----------- SPECIFIC SOC CONSTRAINTS
# =========================================================

class SPHERE_IN: 
    # -- Description: defines a spherical "keep in" region
    def __init__(self,center,radius):
        self.type = 'sphere_in'
        self.convex = True; 
        self.category = 'soc';
        ## General SOC form: ||Cx+d||_2 <= Ax + b;
        self.center = center;
        self.radius = radius;
        self.nx = len(center)
        self.A = np.zeros(self.nx);
        self.b = self.radius; 
        self.C = np.eye(self.nx)
        self.d = - self.center; 
    def g(self,x): return -(mat.norm(x-self.center) - self.radius); 
    def dgdx(self,x): pass # [[TO ADD]]
    def g_jax(self,x): pass # [[TO ADD]]
    def dgdx_jax(self,x): pass # [[TO ADD]]
    def affineApprox(self,x,version='standard'): pass # [[TO ADD]]
    def affineApprox_jax(self,x,version='standard'): pass # [[TO ADD]]
    def cvxApprox(self,x,version='standard'): pass # [[TO ADD]]
    def cvxApprox_jax(self,x,version='standard'): pass # [[TO ADD]]

class SPHERE_OUT:
    # -- Description: defines a spherical keep out region
    def __init__(self,center,radius):
        self.type = 'sphere_out'
        self.convex = False;
        self.category = 'nonconvex';
        self.center = center;
        self.radius = radius; 
    def g(self,x): return mat.norm(x-self.center) - self.radius 
    def dgdx(self,x): pass # [[TO ADD]]
    def g_jax(self,x): pass # [[TO ADD]]
    def dgdx_jax(self,x): pass # [[TO ADD]]
    def affineApprox(self,x,version='standard'):
        diff = self.center - x;
        mag = mat.norm(diff);
        udiff = (1./mag)*diff;
        offset = mag - self.radius; 
        linear = udiff;
        return offset,linear
    def affineApprox_jax(self,x,version='standard'): pass # [[TO ADD]]
    def cvxApprox(self,x,version='standard'): pass # [[TO ADD]]
    def cvxApprox_jax(self,x,version='standard'): pass # [[TO ADD]]


class ELLIPSE_IN:
    ## -- defines a ellipsoidal keep in region
    def __init__(self,center,ins={}):
        self.type = 'ellipse_in'
        self.convex = True;
        self.category = 'soc'
        self.nx = len(center);
        self.center = center;
        self.U = np.eye(nx); self.dists = np.ones(nx)
        if 'U' in ins: self.U = ins['U'];
        if 'dists' in ins: self.dists = ins['dists']
        D = np.diag(self.dists);
        Dinv = np.diag(1./self.dists);
        Uinv = smat.pinv(U); #need to double check... 
        ## General SOC form: ||Cx+d||_2 <= Ax + b;
        self.A = np.zeros(x)
        self.b = 1.; 
        self.C = Dinv@Uinv
        self.d = -Dinv@Uinv@self.center
    def g(self,x): return mat.norm(self.C@x + self.d) - self.A@x - self.b
    def dgdx(self,x): pass # [[TO ADD]]
    def g_jax(self,x): pass # [[TO ADD]]
    def dgdx_jax(self,x): pass # [[TO ADD]]
    def affineApprox(self,x,version='standard'): pass 
    def affineApprox_jax(self,x,version='standard'): pass # [[TO ADD]]
    def cvxApprox(self,x,version='standard'): return self.A,self.b,self.C,self.d
    def cvxApprox_jax(self,x,version='standard'): pass # [[TO ADD]]

class ELLIPSE_OUT: ## NOT FINISHED 
    ## -- defines a keep out ellipsoidal region
    def __init__(self,center,ins={}):
        self.type = 'ellipse_out'
        self.convex = False;
        self.category = 'nonconvex'
        self.nx = len(center);
        self.center = center;
        self.U = np.eye(nx); self.dists = np.ones(nx)
        if 'U' in ins: self.U = ins['U'];
        if 'dists' in ins: self.dists = ins['dists']
        D = np.diag(self.dists);
        Dinv = np.diag(1./self.dists);
        Uinv = smat.pinv(U); #need to double check... 
        #### NEEDS MORE STUFF 
    def g(self,x): return -(mat.norm(self.C@x + self.d) - self.A@x - self.b)
    def dgdx(self,x): pass # [[TO ADD]]
    def g_jax(self,x): pass # [[TO ADD]]
    def dgdx_jax(self,x): pass # [[TO ADD]]
    def affineApprox(self,x,version='standard'): pass 
    def affineApprox_jax(self,x,version='standard'): pass # [[TO ADD]]
    def cvxApprox(self,x,version='standard'): pass
    def cvxApprox_jax(self,x,version='standard'): pass # [[TO ADD]]

class CYLINDER_IN: ## NOT FINISHED 
    def __init__(self,center,ins={}):
        self.type = 'cylinder_in'
        self.convex = True;
        self.category = 'soc'
        self.nx = len(center);
    def g(self,x): pass # [[TO ADD ]]
    def dgdx(self,x): pass # [[TO ADD]]
    def g_jax(self,x): pass # [[TO ADD]]
    def dgdx_jax(self,x): pass # [[TO ADD]]
    def affineApprox(self,x,version='standard'): pass 
    def affineApprox_jax(self,x,version='standard'): pass # [[TO ADD]]
    def cvxApprox(self,x,version='standard'): pass
    def cvxApprox_jax(self,x,version='standard'): pass # [[TO ADD]]


class CYLINDER_OUT: ## NOT FINISHED 
    def __init__(self,center,ins={}):
        self.type = 'cylinder_out'
        self.convex = False;
        self.category = 'nonconvex'
        self.nx = len(center);
    def g(self,x): pass # [[TO ADD ]]
    def dgdx(self,x): pass # [[TO ADD]]
    def g_jax(self,x): pass # [[TO ADD]]
    def dgdx_jax(self,x): pass # [[TO ADD]]
    def affineApprox(self,x,version='standard'): pass 
    def affineApprox_jax(self,x,version='standard'): pass # [[TO ADD]]
    def cvxApprox(self,x,version='standard'): pass
    def cvxApprox_jax(self,x,version='standard'): pass # [[TO ADD]]


class PROX_IN: 
    # defines soc of the form ||x1 - x2 || < r 
    def __init__(self,nx,radius):
        self.type = 'prox_in';
        self.convex = True; 
        self.category = 'soc';
        self.radius = radius;
        self.nx = nx; # dimension of x1 and x2;
        self.A = np.zeros(2*self.nx)
        self.b = self.radius;
        self.C = np.hstack([np.eye(self.nx),-np.eye(self.nx)]);
        self.d = np.zeros(self.nx)
    def g(self,x): pass # [[TO ADD ]]
    def dgdx(self,x): pass # [[TO ADD]]
    def g_jax(self,x): pass # [[TO ADD]]
    def dgdx_jax(self,x): pass # [[TO ADD]]
    def affineApprox(self,x,version='standard'): pass 
    def affineApprox_jax(self,x,version='standard'): pass # [[TO ADD]]
    def cvxApprox(self,x,version='standard'): return self.A,self.b,self.C,self.d
    def cvxApprox_jax(self,x,version='standard'): pass # [[TO ADD]]

class PROX_OUT:
    def __init__(self,radius,nx=None):
        self.type = 'prox_out';
        self.convex = False; 
        self.category = 'soc';
        self.nx = nx;
        self.radius = radius;
    def g(self,x): pass # [[TO ADD ]]
    def dgdx(self,x): pass # [[TO ADD]]
    def g_jax(self,x): pass # [[TO ADD]]
    def dgdx_jax(self,x): pass # [[TO ADD]]
    def affineApprox(self,x,version='standard'):
        # x must have the form x = [x1,x2];
        nx = self.nx; 
        if nx == None: nx = int(0.5*len(x));
        x1 = x[:nx]; x2 = x[nx:]
        z = x1 - x2; 
        center = np.zeros(len(z));
        diff = center - z; 
        mag = mat.norm(diff);
        udiff = (1./mag)*diff;
        Z = np.zeros([nx,int(2*nx)]);
        Z[:,:nx] = np.eye(nx);
        Z[:,nx:] = -np.eye(nx);
        linear = udiff@Z;
        offset = mag - self.radius
        return offset,linear        
    def affineApprox_jax(self,x,version='standard'): pass # [[TO ADD]]
    def cvxApprox(self,x,version='standard'): return self.A,self.b,self.C,self.d
    def cvxApprox_jax(self,x,version='standard'): pass # [[TO ADD]]

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

class CONE_IN: 
    def __init__(self):
        self.type = 'cone_in'
        self.convex = True; 
        self.category = 'soc'
    def g(self,x): pass # [[TO ADD ]]
    def dgdx(self,x): pass # [[TO ADD]]
    def g_jax(self,x): pass # [[TO ADD]]
    def dgdx_jax(self,x): pass # [[TO ADD]]
    def affineApprox(self,x,version='standard'): pass 
    def affineApprox_jax(self,x,version='standard'): pass # [[TO ADD]]
    def cvxApprox(self,x,version='standard'): return # [[TO ADD]]
    def cvxApprox_jax(self,x,version='standard'): pass # [[TO ADD]]        

class CONE_OUT: 
    def __init__(self):
        self.type = 'cone_out'
        self.convex = True; 
        self.category = 'nonconvex'
    def g(self,x): pass # [[TO ADD ]]
    def dgdx(self,x): pass # [[TO ADD]]
    def g_jax(self,x): pass # [[TO ADD]]
    def dgdx_jax(self,x): pass # [[TO ADD]]
    def affineApprox(self,x,version='standard'): pass 
    def affineApprox_jax(self,x,version='standard'): pass # [[TO ADD]]
    def cvxApprox(self,x,version='standard'): return # [[TO ADD]]
    def cvxApprox_jax(self,x,version='standard'): pass # [[TO ADD]]        


class SPHERE_CONE_IN: 
    def __init__(self):
        self.type = 'cone_in'
        self.convex = True; 
        self.category = 'soc'
    def g(self,x): pass # [[TO ADD ]]
    def dgdx(self,x): pass # [[TO ADD]]
    def g_jax(self,x): pass # [[TO ADD]]
    def dgdx_jax(self,x): pass # [[TO ADD]]
    def affineApprox(self,x,version='standard'): pass 
    def affineApprox_jax(self,x,version='standard'): pass # [[TO ADD]]
    def cvxApprox(self,x,version='standard'): return # [[TO ADD]]
    def cvxApprox_jax(self,x,version='standard'): pass # [[TO ADD]]     

class SPHERE_CONE_OUT: 
    def __init__(self):
        self.type = 'cone_out'
        self.convex = True; 
        self.category = 'nonconvex'
    def g(self,x): pass # [[TO ADD ]]
    def dgdx(self,x): pass # [[TO ADD]]
    def g_jax(self,x): pass # [[TO ADD]]
    def dgdx_jax(self,x): pass # [[TO ADD]]
    def affineApprox(self,x,version='standard'): pass 
    def affineApprox_jax(self,x,version='standard'): pass # [[TO ADD]]
    def cvxApprox(self,x,version='standard'): return # [[TO ADD]]
    def cvxApprox_jax(self,x,version='standard'): pass # [[TO ADD]]     

class ELLIPSE_CONE_IN: 
    def __init__(self):
        self.type = 'cone_in'
        self.convex = True; 
        self.category = 'soc'
    def g(self,x): pass # [[TO ADD ]]
    def dgdx(self,x): pass # [[TO ADD]]
    def g_jax(self,x): pass # [[TO ADD]]
    def dgdx_jax(self,x): pass # [[TO ADD]]
    def affineApprox(self,x,version='standard'): pass 
    def affineApprox_jax(self,x,version='standard'): pass # [[TO ADD]]
    def cvxApprox(self,x,version='standard'): return # [[TO ADD]]
    def cvxApprox_jax(self,x,version='standard'): pass # [[TO ADD]]     

class ELLIPSE_CONE_OUT: 
    def __init__(self):
        self.type = 'cone_out'
        self.convex = True; 
        self.category = 'nonconvex'
    def g(self,x): pass # [[TO ADD ]]
    def dgdx(self,x): pass # [[TO ADD]]
    def g_jax(self,x): pass # [[TO ADD]]
    def dgdx_jax(self,x): pass # [[TO ADD]]
    def affineApprox(self,x,version='standard'): pass 
    def affineApprox_jax(self,x,version='standard'): pass # [[TO ADD]]
    def cvxApprox(self,x,version='standard'): return # [[TO ADD]]
    def cvxApprox_jax(self,x,version='standard'): pass # [[TO ADD]]     
