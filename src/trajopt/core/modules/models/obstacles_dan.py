import numpy as np
import scipy as sp
import numpy.linalg as mat
import scipy.linalg as smat

# ==========================================
# -- TYPES OF OBSTACLES TO CONSIDER 
# ==========================================s
# =============================================
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
# ------ specific CONES:
# -- CONE_IN: convex
# -- CONE_OUT: nonconvex approx
# -- SPHERE_CONE_IN: convex
# -- SPHERE_CONE_OUT: nonconvex approx
# -- ELLIPSE_CONE_IN: convex
# -- ELLIPSE_CONE_OUT: nonconvex approx


# =========================================================
# GENERAL LINEAR AFFINE CONSTRAINT 
# =========================================================

class AFFINE: 
    # -- defines affine constraint of the form Ax = b with A = linear and b = offset
    def __init__(self,offset,linear):
        self.type = 'affine'
        self.convex = True; 
        self.category = 'affine'
        self.offset = offset
        self.linear = linear
        self.b = offset
        self.A = linear
    def calcAffine(self):
        return self.offset,self.linear

# =========================================================
# GENERAL POLYTOPE CONSTRAINTS
# =========================================================

class POLYTOPE_IN:
    # -- defines polytope constraint of the form Cx >= d with C = linear and d = offset;
    def __init__(self,offset,linear)
        self.type = 'polytope_in'
        self.convex = True; 
        self.category = 'polytope'
        self.offset = offset
        self.linear = linear
        self.d = offset
        self.C = linear
    def calcPolytope(self):
        return self.offset,self.linear

class POLYTOPE_OUT:  ### NOT FINISHED 
    # -- defines polytope keep out region with polytope Cx >= d
    def __init__(self,offset,linear)
        self.type = 'polytope_out'
        self.convex = False;
        self.category = 'nonconvex'
        self.offset = offset
        self.linear = linear
        self.d = offset
        self.C = linear
    def calcAffine(self,x):
        ## --- needs to be added 
        ## Not obvious how to do this correctly
        pass 

# =========================================================
# ----------- SPECIFIC POLYTOPE CONSTRAINTS
# =========================================================

class ZONOTOPE_IN:  ### NOT FINISHED 
    ## defines convex zonotope constraint
    ## zonotope: a type of polytope constraint with specific structure
    def __init__(self):
        self.type = 'zonotope_in'
        self.convex = True; 
        self.category = 'zonotope'
        ## add more 
class ZONOTOPE_OUT:  ### NOT FINISHED 
    # -- defines zonotope keep out region
    def __init__(self):
        self.type = 'zonotope_out'
        self.convex = False; 
        self.category = 'nonconvex'
        ## add more 


class BOX_IN:
    def __init__(self):
        self.type = 'box_in'
        self.convex = True; 
        self.category = 'box'

class BOX_OUT: 
    def __init__(self):
        self.type = 'box_out'
        self.convex = False; 
        self.category = 'nonconvex'




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
    def calcSOC(self):
        return self.A,self.b,self.C,self.d

class SOC_OUT:     
    def __init__(self,A,b,C,d):
        self.type = 'soc_out'
        self.convex = False;
        self.category = 'nonconvex';
        self.A = A; 
        self.b = b;
        self.C = C;
        self.d = d;
    def calcAffineApprox(self,x):
        pass # need to add


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
    def calcSOC(self):
        return self.A,self.b,self.C,self.d

class SPHERE_OUT:
    # -- Description: defines a spherical keep out region
    def __init__(self,center,radius):
        self.type = 'sphere_out'
        self.convex = False;
        self.category = 'nonconvex';
        self.center = center;
        self.radius = radius; 
    def calcAffineApprox(self,x):
        diff = self.center - x;
        mag = mat.norm(diff);
        udiff = (1./mag)*diff;
        offset = mag - self.radius; 
        linear = udiff;
        return offset,linear


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
    def calcSOC(self):
        return self.A,self.b,self.C,self.d

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
    def calcAffineApprox(self,x,typ='normal'):
        # typ: 'normal' or 'parallel' -- see documentation
        if typ == 'normal':
            pass # need to add 
        if typ == 'parallel':
            pass # need to add 
        offset = None; 
        linear = None; 
        return offset,linear;


class CYLINDER_IN: ## NOT FINISHED 
    def __init__(self,center,ins={}):
        self.type = 'cylinder_in'
        self.convex = True;
        self.category = 'soc'
        self.nx = len(center);

class CYLINDER_OUT: ## NOT FINISHED 
    def __init__(self,center,ins={}):
        self.type = 'cylinder_out'
        self.convex = False;
        self.category = 'nonconvex'
        self.nx = len(center);

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
    def calcSOC(self)
        return self.A,self.b,self.C,self.d

class PROX_OUT:
    def __init__(self,radius,nx=None):
        self.type = 'prox_out';
        self.convex = False; 
        self.category = 'soc';
        self.nx = nx;
        self.radius = radius; 
    def calcAffineApprox(self,x):  ## NEED TO DOUBLE CHECK THIS.  
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




