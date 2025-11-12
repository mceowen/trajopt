import numpy as np
from types import SimpleNamespace

class Indices:
    """
    Generates and stores structured index maps for states, controls,
    dynamics, and constraints inside a Problem.
    """

    def __init__(self, problem):
        self.problem = problem
        self._build_indices()

    def _build_indices(self):
        model   = self.problem.model
        mission = self.problem.mission
        method  = self.problem.method

        # -------------------------------
        # STATE / CONTROL / DYNAMICS INDICES
        # -------------------------------
        n_z   = model.n
        n_u   = model.m
        n_zct = getattr(model, "nz", n_z)   # augmented state
        n_dyn = getattr(mission, "n_dyn", n_zct)

        z_idx_all      = np.arange(0, n_zct)
        z_idx_state    = np.arange(0, n_z)
        z_idx_ctcs     = np.arange(n_z, n_zct) if n_zct > n_z else np.array([], dtype=int)
        z_idx_time     = np.array([], dtype=int)  # optional time variable slot

        self.z = {
            "all":      z_idx_all,
            "state":    z_idx_state,
            "ctcs":     z_idx_ctcs,
            "time":     z_idx_time
        }

        # Controls and other inputs
        u_idx_all = np.arange(0, n_u)
        u_idx_dyn = np.arange(0, n_u)
        u_idx_dilation = np.array([], dtype=int)
        self.nu = {
            "all": u_idx_all,
            "control": u_idx_dyn,
            "dilation_factor": u_idx_dilation
        }

        # -------------------------------
        # CONSTRAINT INDICES
        # -------------------------------
        n_path = mission.n_path
        n_nfz  = mission.n_nfz
        n_aux  = getattr(mission, "n_aux", 0)

        # stacked nonlinear inequality vector
        n_ineq = n_path + n_nfz + n_aux

        # Define slices for subgroups
        i0 = 0
        path_idx = np.arange(i0, i0 + n_path)
        i0 += n_path
        nfz_idx = np.arange(i0, i0 + n_nfz)
        i0 += n_nfz
        aux_idx = np.arange(i0, i0 + n_aux)

        nonlinear_ineq = {
            "all": np.arange(0, n_ineq),
            "path": path_idx,
            "nfz": nfz_idx,
            "aux": aux_idx
        }

        # LTV system / dynamic matrices indices (like your example)
        Ak_ind  = np.arange(0, n_dyn**2)
        Bk_ind  = np.arange(Ak_ind[-1] + 1, Ak_ind[-1] + 1 + n_dyn * model.m)
        Bkp_ind = np.arange(Bk_ind[-1] + 1, Bk_ind[-1] + 1 + n_dyn * model.m)
        Sk_ind  = np.arange(Bkp_ind[-1] + 1, Bkp_ind[-1] + 1 + n_dyn)

        linear_system = {
            "Ak": Ak_ind,
            "Bk": Bk_ind,
            "Bkp": Bkp_ind,
            "Sk": Sk_ind
        }

        self.constraints = SimpleNamespace(
            nonlinear_inequality=nonlinear_ineq,
            linear_system=linear_system
        )

        # -------------------------------
        # Additional CTCS or dynamics couplings
        # -------------------------------
        self.ctcs = {
            "ineq": np.array([], dtype=int),
            "state": z_idx_ctcs
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
