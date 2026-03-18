import numpy as np
import jax.numpy as jnp
from types import SimpleNamespace
from trajopt.utils.tools import AttrDict


class IndexMap:
    """
    Generates and stores structured index maps for states, controls,
    dynamics, and constraints using configs and/or objects.
    """

    def __init__(self, config):
        # Accept configs or objects for flexible initialization
        self.model_config   = config.problem.model
        self.mission_config = config.problem.mission
        self.method_config  = config.method
        self.problem        = None
        self.method         = None

        self.n = AttrDict({
            # core model dims
            "state":        max(max(group["idx"]) for group in self.model_config["state"].values()) + 1,
            "control":      max(max(group["idx"]) for group in self.model_config["control"].values()) + 1,
        })

        self.N = AttrDict({
            # core method dims
            "time_grid":            self.method_config['time_grid']
        })


    def update_index_map(self, problem=None, method=None):
        
        if method is not None and problem is None:
            problem = getattr(method, "problem", None)
        if problem is not None:
            self.problem = problem
        if method is not None:
            self.method = method

        self._build_indices()
    
############################################################
# INDEX PACKING / UNPACKING UTILITIES
############################################################

    def unpack_z(self, z):
        """
        Unpack augmented state z = [x, t, beta].
        """
        x = z[self.indices.z.state]
        t = z[self.indices.z.time]
        beta = z[self.indices.z.ctcs]
        return x, t, beta


    def unpack_nu(self, nu):
        """
        Unpack augmented control nu = [u, s].
        """
        u = nu[self.indices.nu.control]
        s = nu[self.indices.nu.dilation_factor]
        return u, s


    def unpack_znu(self, z, nu):
        """
        Unpack augmented state/control into physically meaningful variables.
        """
        x, t, beta = self.unpack_z(z)
        u, s = self.unpack_nu(nu)
        return x, t, beta, u, s


    def pack_z(self, x, t, beta):
        """
        Pack physical variables into augmented state z = [x, t, beta].
        """
        z = np.zeros(len(self.indices.z.all))
        z[self.indices.z.state] = x
        z[self.indices.z.time] = t
        z[self.indices.z.ctcs] = beta
        return z


    def pack_nu(self, u, s):
        """
        Pack physical variables into augmented control nu = [u, s].
        """
        nu = np.zeros(len(self.indices.nu.all))
        nu[self.indices.nu.control] = u
        nu[self.indices.nu.dilation_factor] = s
        return nu


    def wrap_txu_fcn(self, f_phys):
        """
        Wrap a physical-system function f_phys(t, x, u, params)
        so it can be called as f_alg(z, nu, params).
        """
        def f_alg(z, nu, params):
            x, t, beta, u, s = self.unpack_znu(z, nu)
            return f_phys(t, x, u, params)

        return f_alg


    def wrap_znu_fcn(self, f_alg, beta=None, s=1.0):
        """
        Wrap an algorithm-level function f_alg(z, nu, params)
        so it can be called as f_phys(t, x, u, params).
        """
        def f_phys(t, x, u, params):
            if beta is None:
                beta_vec = np.zeros(len(self.indices.z.ctcs))
            else:
                beta_vec = beta

            z = self.pack_z(x, t, beta_vec)
            nu = self.pack_nu(u, s)
            return f_alg(z, nu, params)

        return f_phys


    def evaluate_f_phys(self, f_phys, z, nu, params):
        """
        Evaluate physical-model function f_phys(t, x, u, params)
        from augmented variables (z, nu).
        """
        x, t, beta, u, s = self.unpack_znu(z, nu)
        return f_phys(t, x, u, params)


    def evaluate_f_alg(self, f_alg, t, x, u, params, beta=None, s=1.0):
        """
        Evaluate algorithm-level function f_alg(z, nu, params)
        from physical variables (t, x, u).
        """
        if beta is None:
            beta = np.zeros(len(self.indices.z.ctcs))

        z = self.pack_z(x, t, beta)
        nu = self.pack_nu(u, s)
        return f_alg(z, nu, params)

############################################################
# BUILD INDICES OBJECTS FROM CONFIGS
############################################################
        
    def _build_indices(self):
        
        # Priority: use configs if provided, else fall back to objects
        if self.model_config is not None:
            nx      = self.n.state
            n_u     = self.n.control
        else:
            nx = n_u = None

        if self.mission_config is not None:
            nz = self.mission_config.get('n_z', nx)
        else:
            nz = nx

        if self.problem is not None and hasattr(self.problem, 'constraints'):
            n_ineq = sum(constraint.dimension for constraint in self.problem.constraints.get(ct=0, type="nonconvex_inequality"))

            if self.problem.constraints.has(ct=1):
                n_ctcs = sum(constraint.dimension for constraint in self.problem.constraints.get(ct=1))
            else:
                n_ctcs = 0

            n_t         = 1
            nz          = nx + n_t + n_ctcs

            n_term      = sum(constraint.dimension for constraint in self.problem.constraints.get(ct=0, type="equality_bc", boundary="final", set="state"))
            n_term_ineq = sum(constraint.dimension for constraint in self.problem.constraints.get(ct=0, type='inequality_bc', boundary="final", set="state"))
            n_term_ctcs = n_ctcs
            n_term_total= n_term + n_term_ineq + n_ctcs
        else:
            n_ineq      = n_path = n_nfz = n_custom = n_ctcs = n_term = n_term_ineq = n_term_ctcs = n_term_total = 0
            n_t         = 1

        n_nu            = n_u + 1

        z_idx_all       = np.arange(0, nz)
        z_idx_state     = np.arange(0, nx)
        z_idx_time      = np.arange(nx, nx + n_t)
        z_idx_ctcs      = np.arange(nx + n_t, nx + n_t + n_ctcs) if n_ctcs > 0 else np.array([], dtype=int)
        nu_idx_all      = np.arange(0, n_nu)
        u_idx_ctrl      = np.arange(0, n_u)
        u_idx_s         = np.arange(n_u, n_u + 1)

        n_term_eq       = n_term
        term_eq_idx     = np.arange(0, n_term_eq)
        term_ineq_idx   = np.arange(n_term_eq, n_term_eq + n_term_ineq)
        term_ctcs_idx   = np.arange(n_term_eq + n_term_ineq, n_term_eq + n_term_ineq + n_term_ctcs)

        Ak_ind          = np.arange(0, nz * nz)
        Bk_ind          = np.arange(Ak_ind[-1] + 1, Ak_ind[-1] + 1 + nz * n_nu)
        Bkp_ind         = np.arange(Bk_ind[-1] + 1, Bk_ind[-1] + 1 + nz * n_nu)

        time_grid       = self.N.time_grid if self.method_config is not None else 0
        buff_dyn        = str(self.method.flags.buff_dyn) if self.method is not None else "l2"
        ctcs            = str(self.method.flags.ctcs) if self.method is not None else "none"

        if buff_dyn in {"term", "l1", "l2"}:
            n_plus_real     = 0
            n_minus_real    = 0
            Npm_real        = 0
        elif buff_dyn == "quad-1":
            n_plus_real     = 1
            n_minus_real    = 1
            Npm_real        = 1
        elif buff_dyn == "quad-2":
            n_plus_real     = 1
            n_minus_real    = 1
            Npm_real        = time_grid - 1
        elif buff_dyn == "quad-3":
            n_plus_real     = nx
            n_minus_real    = nx
            Npm_real        = 1
        else:
            raise ValueError("Invalid buff_dyn flag.")

        if ctcs in {"term", "l1", "l2", "none"}:
            n_plus_ctcs     = 0
            n_minus_ctcs    = 0
            Npm_ctcs        = 0
        elif ctcs == "quad-1":
            n_plus_ctcs     = 1
            n_minus_ctcs    = 1
            Npm_ctcs        = 1
        elif ctcs == "quad-2":
            n_plus_ctcs     = 1
            n_minus_ctcs    = 1
            Npm_ctcs        = time_grid - 1
        elif ctcs == "quad-3":
            n_plus_ctcs     = n_ctcs
            n_minus_ctcs    = n_ctcs
            Npm_ctcs        = 1
        else:
            raise ValueError("Invalid ctcs flag.")

        z_indices = AttrDict({
            "all":          z_idx_all,
            "state":        z_idx_state,
            "time":         z_idx_time,
            "ctcs":         z_idx_ctcs,
        })

        nu_indices = AttrDict({
            "all":              nu_idx_all,
            "control":          u_idx_ctrl,
            "dilation_factor":  u_idx_s,
        })

        nonconvex_inequality = AttrDict({
            "all":          np.arange(0, n_ineq),
        })

        terminal = AttrDict({
            "eq":           term_eq_idx,
            "ineq":         term_ineq_idx,
            "ctcs":         term_ctcs_idx,
            "all":          np.arange(0, n_term_eq + n_term_ineq + n_term_ctcs),
        })

        dynamics_indices = AttrDict({
            "Ak":           Ak_ind,
            "Bk":           Bk_ind,
            "Bkp":          Bkp_ind,
        })

        ctcs_indices = AttrDict({
            "state":        z_idx_ctcs,
            "term":         term_ctcs_idx,
            "ineq":         np.array([], dtype=int),
        })

        self.indices = AttrDict({
            "z":            z_indices,
            "nu":           nu_indices,
            "constraints": AttrDict({
                "nonconvex_inequality": nonconvex_inequality,
                "final_state":          terminal,
                "dynamics":             dynamics_indices,
            }),
            "ctcs":         ctcs_indices,
        })

        self.z_idx      = self.indices.z
        self.nu_idx     = self.indices.nu
        self.ctcs_idx   = self.indices.ctcs

        self.n = AttrDict({
            "state":                int(nx),
            "time":                 int(n_t),
            "control":              int(n_u),
            "nu":                   int(n_nu),
            "z":                    int(nz),
            "ctcs":                 int(n_ctcs),
            "nonconvex_inequality": int(n_ineq),
            "final_state":          int(n_term),
            "term_ineq":            int(n_term_ineq),
            "term_ctcs":            int(n_term_ctcs),
            "term_total":           int(n_term_total),
            "dynamics":             int(nz),
            "plus_real":            int(n_plus_real),
            "minus_real":           int(n_minus_real),
            "plus_ctcs":            int(n_plus_ctcs),
            "minus_ctcs":           int(n_minus_ctcs),
        })

        self.N = AttrDict({
            "nonconvex_inequality": time_grid,
            "dynamics":             time_grid - 1,
            "final_state":          1,
            "box":                  time_grid,
            "control_rate_limit":   time_grid - 1,
            "axis_angle_cone":      time_grid,
            "max_norm_cone":        time_grid,
            "quaternion_cone":      time_grid,
            "AFFINE":               time_grid,
            "POLYTOPE":             time_grid,
            "SOC":                  time_grid,
            "equality_bc":          1,
            "inequality_bc":        1,
            'time_grid':            time_grid,
            "pm_real":              int(Npm_real),
            "pm_ctcs":              int(Npm_ctcs),
        })

    # TODO(Skye): Update this summary
    def summary(self):
        print("==== Problem Indices Summary ====")
        print("\nN (TIME-RELATED INDICES):")
        for k, v in self.N.items():
            print(f"  {k:25s}: {v}")
        
        print("\nINDEX ARRAYS:")
        for obj_name in ['z', 'nu', 'ctcs']:
            obj = self.indices[obj_name]
            print(f"  {obj_name.upper()}:")
            for k, v in obj.items():
                print(f"    {k:25s}: {v.shape if hasattr(v,'shape') else len(v)}")

        print("\nCONSTRAINTS:")
        for constraint_type in ['nonlinear_inequality', 'final_state', 'dynamics']:
            constraint_obj = getattr(self.indices.constraints, constraint_type)
            print(f"    {constraint_type.upper()}:")
            for k, v in constraint_obj.items():
                print(f"      {k:25s}: {v.shape if hasattr(v,'shape') else len(v)}")

        print("\nSCALAR COUNTS (self.n):")
        for k, v in self.n.items():
            print(f"  {k:25s}: {v}")

