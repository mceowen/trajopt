import numpy as np
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
            "state":        self.model_config['dimensions']['n'],
            "control":      self.model_config['dimensions']['m'],
        })
        self.N = AttrDict({
            "N":            self.method_config["N"]
        })


    def update_index_map(self, problem=None, method=None):
        if method is not None and problem is None:
            problem = getattr(method, "problem", None)
        if problem is not None:
            self.problem = problem
        if method is not None:
            self.method = method
        self._build_indices()
        

    def _build_indices(self):
        # Priority: use configs if provided, else fall back to objects
        if self.model_config is not None:
            nx      = self.model_config['dimensions']['n']
            n_nu    = self.model_config['dimensions']['m']
        else:
            nx = n_nu = None

        if self.mission_config is not None:
            nz = self.mission_config.get('n_z', nx)
        else:
            nz = nx

        if self.problem is not None and hasattr(self.problem, 'constraints'):
            n_ineq = sum(constraint.dimension for constraint in self.problem.constraints.get(ct=0, type="nonconvex_inequality"))
            n_path = sum(constraint.dimension for constraint in self.problem.constraints.get(ct=0, type="nonconvex_inequality", group="path"))
            n_nfz = sum(constraint.dimension for constraint in self.problem.constraints.get(ct=0, type="nonconvex_inequality", group="nfz"))
            n_custom = sum(constraint.dimension for constraint in self.problem.constraints.get(ct=0, type="nonconvex_inequality", group="custom"))

            if self.problem.constraints.has(ct=1):
                n_ctcs = sum(constraint.dimension for constraint in self.problem.constraints.get(ct=1))
            else:
                n_ctcs = 0

            nz          = nx + n_ctcs

            n_term      = sum(constraint.dimension for constraint in self.problem.constraints.get(ct=0, type="equality_bc", boundary="final", set="state"))
            n_term_ineq = sum(constraint.dimension for constraint in self.problem.constraints.get(ct=0, type='inequality_bc', boundary="final", set="state"))
            n_term_ctcs = n_ctcs
            n_term_total= n_term + n_term_ineq + n_ctcs
        else:
            n_ineq      = n_path = n_nfz = n_custom = n_ctcs = n_term = n_term_ineq = n_term_ctcs = n_term_total = 0

        z_idx_all   = np.arange(0, nz)
        z_idx_state = np.arange(0, nx)
        z_idx_ctcs  = np.arange(nx, nx + n_ctcs) if n_ctcs > 0 else np.array([], dtype=int)
        u_idx_all   = np.arange(0, n_nu)

        i0 = 0
        path_idx    = np.arange(i0, i0 + n_path)
        i0 += n_path
        nfz_idx     = np.arange(i0, i0 + n_nfz)
        i0 += n_nfz
        custom_idx  = np.arange(i0, i0 + n_custom)

        n_term_eq   = n_term
        eq_idx      = np.arange(0, n_term_eq)
        ineq_idx    = np.arange(n_term_eq, n_term_eq + n_term_ineq)
        ctcs_idx    = np.arange(n_term_eq + n_term_ineq, n_term_eq + n_term_ineq + n_term_ctcs)

        Ak_ind      = np.arange(0, nz * nz)
        Bk_ind      = np.arange(Ak_ind[-1] + 1, Ak_ind[-1] + 1 + nz * n_nu)
        Bkp_ind     = np.arange(Bk_ind[-1] + 1, Bk_ind[-1] + 1 + nz * n_nu)
        Sk_ind      = np.arange(Bkp_ind[-1] + 1, Bkp_ind[-1] + 1 + nz)

        N_val = self.method_config['N'] if self.method_config is not None else 0
        buff_dyn = str(self.method.flags.buff_dyn) if self.method is not None else "term"
        ctcs = str(self.method.flags.ctcs) if self.method is not None else "none"

        if buff_dyn in {"term", "l1", "l2"}:
            n_plus_real = 0
            n_minus_real = 0
            Npm_real = 0
        elif buff_dyn == "quad-1":
            n_plus_real = 1
            n_minus_real = 1
            Npm_real = 1
        elif buff_dyn == "quad-2":
            n_plus_real = 1
            n_minus_real = 1
            Npm_real = N_val - 1
        elif buff_dyn == "quad-3":
            n_plus_real = nx
            n_minus_real = nx
            Npm_real = 1
        else:
            raise ValueError("Invalid buff_dyn flag.")

        if ctcs in {"term", "l1", "l2", "none"}:
            n_plus_ctcs = 0
            n_minus_ctcs = 0
            Npm_ctcs = 0
        elif ctcs == "quad-1":
            n_plus_ctcs = 1
            n_minus_ctcs = 1
            Npm_ctcs = 1
        elif ctcs == "quad-2":
            n_plus_ctcs = 1
            n_minus_ctcs = 1
            Npm_ctcs = N_val - 1
        elif ctcs == "quad-3":
            n_plus_ctcs = n_ctcs
            n_minus_ctcs = n_ctcs
            Npm_ctcs = 1
        else:
            raise ValueError("Invalid ctcs flag.")

        z_indices = AttrDict({
            "all": z_idx_all,
            "state": z_idx_state,
            "ctcs": z_idx_ctcs,
        })

        nu_indices = AttrDict({
            "all": u_idx_all,
            "control": u_idx_all,
            "dilation_factor": np.array([], dtype=int),
        })

        nonlinear_ineq = AttrDict({
            "all": np.arange(0, n_ineq),
            "path": path_idx,
            "nfz": nfz_idx,
            "custom": custom_idx,
        })

        terminal = AttrDict({
            "eq": eq_idx,
            "ineq": ineq_idx,
            "ctcs": ctcs_idx,
            "all": np.arange(0, n_term_eq + n_term_ineq + n_term_ctcs),
        })

        dynamics_indices = AttrDict({
            "Ak": Ak_ind,
            "Bk": Bk_ind,
            "Bkp": Bkp_ind,
            "Sk": Sk_ind,
        })

        ctcs_indices = AttrDict({
            "state": z_idx_ctcs,
            "term": ctcs_idx,
            "ineq": np.array([], dtype=int),
        })

        self.indices = AttrDict({
            "z": z_indices,
            "nu": nu_indices,
            "constraints": AttrDict({
                "nonlinear_inequality": nonlinear_ineq,
                "terminal": terminal,
                "dynamics": dynamics_indices,
            }),
            "ctcs": ctcs_indices,
        })

        self.n = AttrDict({
            "state": int(nx),
            "control": int(n_nu),
            "z": int(nz),
            "ctcs": int(n_ctcs),
            "path": int(n_path),
            "nfz": int(n_nfz),
            "custom": int(n_custom),
            "nonconvex_inequality": int(n_ineq),
            "terminal": int(n_term),
            "term_ineq": int(n_term_ineq),
            "term_ctcs": int(n_term_ctcs),
            "term_total": int(n_term_total),
            "dynamics": int(nz),
            "plus_real": int(n_plus_real),
            "minus_real": int(n_minus_real),
            "plus_ctcs": int(n_plus_ctcs),
            "minus_ctcs": int(n_minus_ctcs),
        })

        self.N = AttrDict({
            "nonconvex_inequality": N_val,
            "path": N_val,
            "nfz": N_val,
            "custom": N_val,
            "dynamics": N_val - 1,
            "terminal": 1,
            "box": N_val,
            "control_rate_limit": N_val - 1,
            "axis_angle_cone": N_val,
            "max_norm_cone": N_val,
            "quaternion_cone": N_val,
            "AFFINE": N_val,
            "POLYTOPE": N_val,
            "SOC": N_val,
            "equality_bc": 1,
            "inequality_bc": 1,
            "N": N_val,
            "pm_real": int(Npm_real),
            "pm_ctcs": int(Npm_ctcs),
        })

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
        for constraint_type in ['nonlinear_inequality', 'terminal', 'dynamics']:
            constraint_obj = getattr(self.indices.constraints, constraint_type)
            print(f"    {constraint_type.upper()}:")
            for k, v in constraint_obj.items():
                print(f"      {k:25s}: {v.shape if hasattr(v,'shape') else len(v)}")

        print("\nSCALAR COUNTS (self.n):")
        for k, v in self.n.items():
            print(f"  {k:25s}: {v}")

