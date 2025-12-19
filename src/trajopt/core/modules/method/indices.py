import numpy as np
from types import SimpleNamespace

class Indices:
    """
    Generates and stores structured index maps for states, controls,
    dynamics, and constraints inside a Problem.
    """

    def __init__(self, trajopt_obj):
        self.trajopt_obj = trajopt_obj
        self._build_indices()

    def _build_indices(self):
        model   = self.trajopt_obj.model
        mission = self.trajopt_obj.mission

        # -------------------------------------------------
        # STATE / CONTROL / CTCS INDICES
        # -------------------------------------------------
        n      = model.n
        nz     = model.nz
        n_ctcs = model.n_ctcs

        z_idx_all   = np.arange(0, nz)
        z_idx_state = np.arange(0, n)
        z_idx_ctcs  = np.arange(n, n + n_ctcs) if n_ctcs > 0 else np.array([], dtype=int)

        self.z = {
            "all":    z_idx_all,
            "state":  z_idx_state,
            "ctcs":   z_idx_ctcs,
            "time":   np.array([], dtype=int),
        }

        # -------------------------------------------------
        # CONTROL INDICES
        # -------------------------------------------------
        u_idx_all = np.arange(0, model.m)
        self.nu = {
            "all": u_idx_all,
            "control": u_idx_all,
            "dilation_factor": np.array([], dtype=int),
        }

        # -------------------------------------------------
        # NONLINEAR INEQUALITY INDICES
        # -------------------------------------------------
        n_path   = mission.n_path
        n_nfz    = mission.n_nfz
        n_custom = mission.n_custom
        n_ineq   = mission.n_ineq

        i0 = 0
        path_idx   = np.arange(i0, i0 + n_path);   i0 += n_path
        nfz_idx    = np.arange(i0, i0 + n_nfz);    i0 += n_nfz
        custom_idx = np.arange(i0, i0 + n_custom)

        nonlinear_ineq = {
            "all":    np.arange(0, n_ineq),
            "path":   path_idx,
            "nfz":    nfz_idx,
            "custom": custom_idx,
        }

        # -------------------------------------------------
        # TERMINAL CONDITION INDICES (EQ + INEQ + CTCS)
        # -------------------------------------------------
        n_term_eq   = mission.n_term            # terminal eq constraints
        n_term_ineq = mission.n_term_ineq       # terminal inequalities
        n_term_ctcs = mission.n_term_ctcs       # CTCS terminal constraints

        eq_idx   = np.arange(0, n_term_eq)
        ineq_idx = np.arange(n_term_eq,
                            n_term_eq + n_term_ineq)
        ctcs_idx = np.arange(n_term_eq + n_term_ineq,
                            n_term_eq + n_term_ineq + n_term_ctcs)

        terminal = {
            "eq":   eq_idx,
            "ineq": ineq_idx,
            "ctcs": ctcs_idx,
            "all":  np.arange(0, n_term_eq + n_term_ineq + n_term_ctcs),
        }

        # -------------------------------------------------
        # LTV SYSTEM INDICES
        # -------------------------------------------------
        Ak_ind  = np.arange(0, nz * nz)
        Bk_ind  = np.arange(Ak_ind[-1] + 1, Ak_ind[-1] + 1 + nz * model.m)
        Bkp_ind = np.arange(Bk_ind[-1] + 1, Bk_ind[-1] + 1 + nz * model.m)
        Sk_ind  = np.arange(Bkp_ind[-1] + 1, Bkp_ind[-1] + 1 + nz)

        linear_system = {
            "Ak":  Ak_ind,
            "Bk":  Bk_ind,
            "Bkp": Bkp_ind,
            "Sk":  Sk_ind,
        }

        # -------------------------------------------------
        # Populate namespace
        # -------------------------------------------------
        from types import SimpleNamespace
        self.constraints = SimpleNamespace(
            nonlinear_inequality = nonlinear_ineq,
            terminal             = terminal,
            linear_system        = linear_system,
        )

        # CTCS indexing
        self.ctcs = {
            "state": z_idx_ctcs,
            "term":  ctcs_idx,
            "ineq":  np.array([], dtype=int),
        }

    def summary(self):
        print("==== Problem Indices Summary ====")
        for group_name, group in vars(self).items():
            if isinstance(group, dict):
                print(f"\n{group_name.upper()}:")
                for k, v in group.items():
                    print(f"  {k:25s}: {v.shape if hasattr(v,'shape') else len(v)}")
            elif isinstance(group, SimpleNamespace):
                print(f"\n{group_name.upper()}:")
                for k, v in vars(group).items():
                    print(f"  {k:25s}: {v.shape if hasattr(v,'shape') else len(v)}")
