from typing import Dict, Any, Optional, List
import time
import numpy as np
import cvxpy as cp
import jax.numpy as jnp
import copy

from trajopt.library.methods.subproblem_constraints import SubproblemConstraints
from trajopt.library.methods import discretize
from trajopt.library.methods import convergence
from trajopt.library.methods import hyperparameters as hp
from trajopt.utils import tools

# =========================
# Subproblem (build-once)
# =========================
class Subproblem:
    """Reusable convex SCP with full baseline functionality & DPP updates.
    """
    def __init__(self, problem, method) -> None:
        self.problem = problem
        self.method  = method

        # subproblem constraints
        self.constraints = SubproblemConstraints(problem, method)

        # derive canonical sizes for quick reuse
        self.N                  = int(method.N)
        self.n                  = int(problem.n)
        self.m                  = int(problem.m)
        self.nz                 = int(problem.nz)
        self.n_ctcs             = int(problem.n_ctcs)
        self.n_path             = int(problem.n_path)
        self.n_nfz              = int(problem.n_nfz)
        self.n_custom           = int(problem.n_custom)
        self.n_ineq             = int(problem.n_ineq)
        self.n_term             = int(problem.n_term)
        self.n_term_ineq        = int(problem.n_term_ineq)
        self.n_term_ctcs        = int(problem.n_term_ctcs)
        self.n_term_total       = int(problem.n_term_total)
        self.n_dyn              = int(problem.nz)
        self.Npm_real           = int(getattr(method, "Npm_real", 0))
        self.n_plus_real        = int(getattr(method, "n_plus_real", 0))
        self.n_minus_real       = int(getattr(method, "n_minus_real", 0))
        self.Npm_ctcs           = int(getattr(method, "Npm_ctcs", 0))
        self.n_plus_ctcs        = int(getattr(method, "n_plus_ctcs", 0))
        self.n_minus_ctcs       = int(getattr(method, "n_minus_ctcs", 0))

        # Optional module flags as Parameters (enable gating)
        self.flags          = method.flags
        self.free_T         = bool(self.flags["free_final_time"])
        self.equal_dt       = bool(self.flags["equal_dt"])
        self.buff_dyn       = self.flags["buff_dyn"]  
        self.ctcs           = self.flags["ctcs"]         # e.g., "l1", "l2", "term", "quad-1", "quad-2"
        self.flag_autotune  = self.flags["flag_autotune"]
        self.flag_path      = method.flags.get("flag_path", 1.0)
        self.flag_nfz       = method.flags.get("flag_nfz", 1.0)
        self.flag_custom    = method.flags.get("flag_custom", 1.0)
        self.flag_term      = method.flags.get("flag_term", 1.0)
        self.flag_dyn       = method.flags.get("flag_dyn", 1.0)
        self.flag_tr        = method.flags.get("flag_tr", 1.0)
        self.flag_true      = method.flags.get("flag_true", 1.0)
        self.flag_dual      = method.flags.get("flag_dual", 1.0)
        self.flag_vb        = method.flags.get("flag_vb", 1.0)

        # Build the DPP graph once
        self._create_variables()
        self._create_parameters()
        self.cp_constraints: List[cp.constraints.constraint.Constraint] = []
        self._build_constraints_once()
        self.cost_expr = self._build_cost_once()

        # apply custom constraints and cost
        # mission.custom_constraints(self)
        # mission.custom_cost(self)

        # Compile CVXPY problem once
        self.subproblem = cp.Problem(cp.Minimize(self.cost_expr), self.cp_constraints)

        total_param_scalars = sum(p.size for p in self.subproblem.parameters())
        print(f"total number of parameters: {total_param_scalars}")

        # --------------------------
        # Initialize unified history
        # --------------------------
        self.iter_data: List[Dict[str, Any]] = [{
            "iter_num": 0,  # init only (no outputs yet)
            "z_ref": method.z_init,
            "nu_ref": method.nu_init,
            "dt_ref": method.dt_init,
            "t_ref": method.t_init,
            "conv_data": {
                "vb_ineq": np.zeros((self.N, problem.n_ineq)),
                "vb_dyn":  np.zeros((self.N - 1, self.n_dyn)),
                "vb_term": np.zeros((problem.n_term_total, 1)),
            },
            "weights": copy.deepcopy(method.weights),
        }]

    # ============================================================
    # VARIABLE & PARAMETER CREATION
    # ============================================================
    def _create_variables(self) -> None:
        problem, method = self.problem, self.method
        N, n, m, nz, n_ctcs = self.N, self.n, self.m, self.nz, self.n_ctcs

        # Core optimization variables
        self.dz = cp.Variable((N, nz), name="dz")
        self.dnu= cp.Variable((N, m), name="du")

        # Time variable(s)
        if self.free_T:
            if self.equal_dt:
                self.dT = cp.Variable(name="dT")
                self.dt = (1 / (N - 1)) * self.dT * np.ones((N - 1, 1))
            else:
                self.dT = None
                self.dt = cp.Variable((N - 1, 1), name="dt_opt")
        else:
            self.dT = None
            self.dt = cp.Constant(np.zeros((N - 1, 1)))  # CVXPY constant, safe in constraints


        # Virtual buffers (None if zero-sized)
        self.vb_ineq    = cp.Variable((N, problem.n_ineq), name="vb_ineq")   if problem.n_ineq  > 0 else None 
        
        # ---------------------------------------------
        # TERMINAL CONDITION BUFFERS (REAL + CTCS)
        # --- ------------------------------------------
        n_term_real = self.n_term + self.n_term_ineq
        n_term_ctcs  = self.n_term_ctcs
        # ------------ Real terminal constraints (size = n_term_real) ------------
        self.vb_term_real = (
            cp.Variable(n_term_real, name="vb_term_real")
            if (method.flags["buff_dyn"] == "term"
                and method.flags["dynamics_nonconvex"] != 0
                and n_term_real > 0)
            else (cp.Constant(np.zeros(n_term_real)) if n_term_real > 0 else None)
        )
        # ------------ CTCS terminal constraints (size = n_term_ctcs) ------------
        self.vb_term_ctcs = (
            cp.Variable(n_term_ctcs, name="vb_term_ctcs")
            if (n_term_ctcs > 0)  
            else None
        )
        # ------------ Unified stacked terminal buffer ------------
        self.vb_term = (
            cp.hstack([self.vb_term_real, self.vb_term_ctcs])
            if (self.vb_term_real is not None and self.vb_term_ctcs is not None)
            else self.vb_term_real
            if self.vb_term_ctcs is None
            else self.vb_term_ctcs
            if self.vb_term_real is None
            else None # when both are None
        )

        # ---------------------------------------------
        # DYNAMICS VIRTUAL BUFFERS (REAL + CTCS)
        # ---------------------------------------------
        # --- Physical dynamics (first n states) ---
        self.vb_dyn_real_p = (
            cp.Variable((N - 1, n), name="vb_dyn_real_plus")
            if method.flags["buff_dyn"] != "term" and method.flags["dynamics_nonconvex"] != 0 and n > 0
            else cp.Constant(np.zeros((N - 1, n))) if n > 0
            else None
        )
        self.vb_dyn_real_m = (
            cp.Variable((N - 1, n), name="vb_dyn_real_minus")
            if method.flags["buff_dyn"] != "term" and method.flags["dynamics_nonconvex"] != 0 and n > 0
            else cp.Constant(np.zeros((N - 1, n))) if n > 0
            else None
        )
        # --- CTCS dynamics (augmented states n : nz) ---
        self.vb_dyn_ctcs_p = (
            cp.Variable((N - 1, n_ctcs), name="vb_dyn_ctcs_plus")
            if method.flags["ctcs"] not in {"none", "term"} and n_ctcs > 0
            else cp.Constant(np.zeros((N - 1, n_ctcs))) if n_ctcs > 0
            else None
        )
        self.vb_dyn_ctcs_m = (
            cp.Variable((N - 1, n_ctcs), name="vb_dyn_ctcs_minus")
            if method.flags["ctcs"] not in {"none", "term"} and n_ctcs > 0
            else cp.Constant(np.zeros((N - 1, n_ctcs))) if n_ctcs > 0
            else None
        )
        # --- Unified composite buffers (always same shape for DPP) ---
        self.vb_dyn_p = (
            cp.hstack([self.vb_dyn_real_p, self.vb_dyn_ctcs_p])
            if n_ctcs>0 else self.vb_dyn_real_p
        )
        self.vb_dyn_m = (
            cp.hstack([self.vb_dyn_real_m, self.vb_dyn_ctcs_m])
            if n_ctcs>0 else self.vb_dyn_real_m
        )

        # Aggregate buffers (optional)
        self.vb_plus_real  = cp.Variable((self.Npm_real, self.n_plus_real),  name="vb_plus_real")  if self.n_plus_real  > 0 else None
        self.vb_minus_real = cp.Variable((self.Npm_real, self.n_minus_real), name="vb_minus_real") if self.n_minus_real > 0 else None
        self.vb_plus_ctcs  = cp.Variable((self.Npm_ctcs, self.n_plus_ctcs),  name="vb_plus_ctcs")  if self.n_plus_ctcs  > 0 else None
        self.vb_minus_ctcs = cp.Variable((self.Npm_ctcs, self.n_minus_ctcs), name="vb_minus_ctcs") if self.n_minus_ctcs > 0 else None

    def _create_parameters(self) -> None:
        problem, method = self.problem, self.method
        N, n, m, nz = self.N, self.n, self.m, self.nz

        # Linearized dynamics & trajectory references
        self.Ak     = cp.Parameter((N - 1, nz, nz), name="Ak")
        self.Bk     = cp.Parameter((N - 1, nz, m), name="Bk")
        self.Bkp    = cp.Parameter((N - 1, nz, m), name="Bkp")
        self.Sk     = cp.Parameter((N - 1, nz),    name="Sk")
        self.z_m   = cp.Parameter((N, nz),        name="z_minus")

        self.w_cost_times_dcostdz = cp.Parameter((N, n), name="dcostdz")
        self.w_cost_times_dcostdnu = cp.Parameter((N, m), name="dcostdnu")
        self.w_cost_times_cost0   = cp.Parameter(N,      name="cost0")

        self.z_ref      = cp.Parameter((N, nz),    name="z_ref")
        self.nu_ref     = cp.Parameter((N, m),    name="nu_ref")
        self.dt_ref     = cp.Parameter((N - 1, 1),name="dt_ref", nonneg=True)

        self.nu_ref_sq  = cp.Parameter((N,), name="nu_ref_sq")

        # Path/NFZ/AUX linearized constraints
        if problem.constraints.get("nodal", "nonconvex_inequality"):
            self.dgdz   = cp.Parameter((N, problem.n_ineq, n), name="dgdz")
            self.dgdnu  = cp.Parameter((N, problem.n_ineq, m), name="dgdnu")
            self.g0     = cp.Parameter((N, problem.n_ineq),    name="g0")
        else:
            self.dgdz = self.dgdnu = self.g0 = None

        # time-step scalar bounds
        if self.free_T:
            self.dt_min  = cp.Parameter(nonneg=True, name="dt_min")
            self.dt_max  = cp.Parameter(nonneg=True, name="dt_max")
            self.ddt_max = cp.Parameter(nonneg=True, name="ddt_max")
        else:
            self.dt_min = self.dt_max = self.ddt_max = None

        # Weights & Trust Region weights
        self.w_cost = cp.Parameter(nonneg=True, name="w_cost")
        self.wtr_z  = cp.Parameter(nonneg=True, name="wtr_z")
        self.wtr_u  = cp.Parameter(nonneg=True, name="wtr_u")

        # TODO(Skye): refactor these weight parameters to be less redundant later
        self.W_ineq  = cp.Parameter((N,  max(self.n_ineq, 1)),  nonneg=True, name="W_ineq")
        self.W_term  = cp.Parameter((max(self.n_term_total, 1),),     nonneg=True, name="W_term")
        self.W_dyn   = cp.Parameter((N - 1, max(nz, 1)),        nonneg=True, name="W_dyn")

        self.W_plus_real  = cp.Parameter((max(self.Npm_real, 1), max(self.n_plus_real, 1)),  nonneg=True, name="W_plus_real")
        self.W_minus_real = cp.Parameter((max(self.Npm_real, 1), max(self.n_minus_real, 1)), nonneg=True, name="W_minus_real")
        self.W_plus_ctcs  = cp.Parameter((max(self.Npm_ctcs, 1), max(self.n_plus_ctcs, 1)),  nonneg=True, name="W_plus_ctcs")
        self.W_minus_ctcs = cp.Parameter((max(self.Npm_ctcs, 1), max(self.n_minus_ctcs, 1)), nonneg=True, name="W_minus_ctcs")
        # ----------------------------------------------------------------------------------------------------------
        self.W_ineq_sqrt  = cp.Parameter((N,  max(self.n_ineq, 1)),  nonneg=True, name="W_ineq_sqrt")
        self.W_term_sqrt  = cp.Parameter((max(self.n_term_total, 1),),     nonneg=True, name="W_term_sqrt")
        self.W_dyn_sqrt   = cp.Parameter((N - 1, max(nz, 1)),        nonneg=True, name="W_dyn_sqrt")
        
        self.W_plus_real_sqrt  = cp.Parameter((max(self.Npm_real, 1), max(self.n_plus_real, 1)),  nonneg=True, name="W_plus_real_sqrt")
        self.W_minus_real_sqrt = cp.Parameter((max(self.Npm_real, 1), max(self.n_minus_real, 1)), nonneg=True, name="W_minus_real_sqrt")
        self.W_plus_ctcs_sqrt  = cp.Parameter((max(self.Npm_ctcs, 1), max(self.n_plus_ctcs, 1)),  nonneg=True, name="W_plus_ctcs_sqrt")
        self.W_minus_ctcs_sqrt = cp.Parameter((max(self.Npm_ctcs, 1), max(self.n_minus_ctcs, 1)), nonneg=True, name="W_minus_ctcs_sqrt")
        # ----------------------------------------------------------------------------------------------------------
        self.w_ineq_row = cp.Parameter(N,  nonneg=True, name="w_ineq_row")  if self.n_ineq > 0 else None
        self.w_dyn_row  = cp.Parameter(N - 1, nonneg=True, name="w_dyn_row")
        # ----------------------------------------------------------------------------------------------------------
        self.W_ineq.value  = np.ones((N,  max(self.n_ineq, 1)))
        self.W_term.value  = np.ones((max(self.n_term_total, 1),))
        self.W_dyn.value   = np.ones((N - 1, max(nz, 1)))
        self.W_plus_real.value  = np.ones((max(self.Npm_real, 1), max(self.n_plus_real, 1)))
        self.W_minus_real.value = np.ones((max(self.Npm_real, 1), max(self.n_minus_real, 1)))
        self.W_plus_ctcs.value  = np.ones((max(self.Npm_ctcs, 1), max(self.n_plus_ctcs, 1)))
        self.W_minus_ctcs.value = np.ones((max(self.Npm_ctcs, 1), max(self.n_minus_ctcs, 1)))
        # ----------------------------------------------------------------------------------------------------------
        self.W_ineq_sqrt.value  = np.sqrt(self.W_ineq.value)
        self.W_term_sqrt.value  = np.sqrt(self.W_term.value)
        self.W_dyn_sqrt.value   = np.sqrt(self.W_dyn.value)
        self.W_plus_real_sqrt.value  = np.sqrt(self.W_plus_real.value)
        self.W_minus_real_sqrt.value = np.sqrt(self.W_minus_real.value)
        self.W_plus_ctcs_sqrt.value  = np.sqrt(self.W_plus_ctcs.value)
        self.W_minus_ctcs_sqrt.value = np.sqrt(self.W_minus_ctcs.value)
        # ----------------------------------------------------------------------------------------------------------
        if self.w_ineq_row is not None:
            self.w_ineq_row.value = np.max(self.W_ineq.value, axis=1)
        self.w_dyn_row.value = np.max(self.W_dyn.value, axis=1)

        # duals (same ≥1-column pattern, unified inequality structure)
        self.dual_ineq  = cp.Parameter((N,  max(self.n_ineq, 1)), name="dual_ineq")
        self.dual_dyn   = cp.Parameter((N - 1, nz),              name="dual_dyn")
        
        self.dual_plus_real  = cp.Parameter((max(self.Npm_real, 1), max(self.n_plus_real, 1)),  name="dual_plus_real")
        self.dual_minus_real = cp.Parameter((max(self.Npm_real, 1), max(self.n_minus_real, 1)), name="dual_minus_real")
        self.dual_plus_ctcs  = cp.Parameter((max(self.Npm_ctcs, 1), max(self.n_plus_ctcs, 1)),  name="dual_plus_ctcs")
        self.dual_minus_ctcs = cp.Parameter((max(self.Npm_ctcs, 1), max(self.n_minus_ctcs, 1)), name="dual_minus_ctcs")
        
        self.dual_term  = cp.Parameter((max(self.n_term_total, 1),),                  name="dual_term")

        # CTCS epsilon (scalar)
        self.eps_ctcs = cp.Parameter(nonneg=True, name="eps_ctcs")

    # ============================================================
    # CONSTRAINTS (build-once)
    # ============================================================
    def _build_constraints_once(self) -> None:
        problem     = self.problem
        method      = self.method
        index_map   = method.index_map

        N, n, m, nz, n_ctcs     = self.N, self.n, self.m, self.nz, self.n_ctcs

        C: List[cp.Constraint] = []

        # Terminal equalities / inequalities
        term_idx  = index_map.constraints.terminal  

        for constraint in problem.constraints.get("nodal", "equality_bc"):
            x_idx = constraint.x_idx
            idx   = constraint.idx
            x     = constraint.x

            if constraint.set == "state":
                if constraint.boundary == "final":
                    vb = self.vb_term[term_idx["eq"]] if self.vb_term is not None else 0.0
                else:
                    vb = 0
                C.append(self.dz[idx,x_idx] + self.z_ref[idx, x_idx] - vb == x)
            elif constraint.set == "control":
                C.append(self.dnu[idx,x_idx] + self.nu_ref[idx, x_idx] == x)

        for constraint in problem.constraints.get("nodal", "inequality_bc"):

            x_min_idx = constraint.x_min_idx
            x_max_idx = constraint.x_max_idx
            x_min = constraint.x_min
            x_max = constraint.x_max
            idx   = constraint.idx
            M_select = constraint.M_select

            if constraint.set == "state":
                if constraint.boundary == "final":
                    vb = self.vb_term[term_idx["ineq"]] if self.vb_term is not None else 0.0
                else:
                    vb = 0
                
                C.append(M_select @ (self.dz[idx, :n] + self.z_ref[idx, :n]) - vb <= cp.hstack([-x_min, x_max]))
            
            elif constraint.set == "control":
                C.append(M_select @ (self.dnu[idx, :m] + self.nu_ref[idx, :m]) <= cp.hstack([-x_min, x_max]))
        
        # CTCS terminal equalities
        if problem.n_term_ctcs>0:
            ctcs_state_idx = index_map.z["ctcs"]  
            vbN_ctcs = self.vb_term[term_idx["ctcs"]] if self.vb_term is not None else 0.0
            C.append(
                self.dz[-1, ctcs_state_idx] + self.z_ref[-1, ctcs_state_idx] - vbN_ctcs == 0.0
            )
        
        if self.buff_dyn == "quad-1":
            C.append(cp.sum(self.vb_dyn_p) == self.vb_plus_real)
            C.append(cp.sum(self.vb_dyn_m) == self.vb_minus_real)

        if self.ctcs == "quad-1":
            C.append(cp.sum(self.vb_dyn_p) == self.vb_plus_ctcs)
            C.append(cp.sum(self.vb_dyn_m) == self.vb_minus_ctcs)

        if self.buff_dyn == "quad-2":
            C.append(cp.sum(self.vb_dyn_p[:, index_map.z["state"]], axis=1) == self.vb_plus_real[:, 0])
            C.append(cp.sum(self.vb_dyn_m[:, index_map.z["state"]], axis=1) == self.vb_minus_real[:, 0])

        if self.ctcs == "quad-2":
            C.append(cp.sum(self.vb_dyn_p[:, index_map.z["ctcs"]], axis=1) == self.vb_plus_ctcs[:, 0])
            C.append(cp.sum(self.vb_dyn_m[:, index_map.z["ctcs"]], axis=1) == self.vb_minus_ctcs[:, 0])

        if self.buff_dyn == "quad-3":
            C.append(cp.sum(self.vb_dyn_p[:, index_map.z["state"]], axis=0) == self.vb_plus_real[0, :])
            C.append(cp.sum(self.vb_dyn_m[:, index_map.z["state"]], axis=0) == self.vb_minus_real[0, :])
        
        if self.ctcs == "quad-3":
            C.append(cp.sum(self.vb_dyn_p[:, index_map.z["ctcs"]], axis=0) == self.vb_plus_ctcs[0, :])
            C.append(cp.sum(self.vb_dyn_m[:, index_map.z["ctcs"]], axis=0) == self.vb_minus_ctcs[0, :])

        # Per-stage constraints
        for k in range(N):
            if k < N - 1:
                # Discrete dynamics with dyn buffers
                rhs = (
                    self.Ak[k] @ self.dz[k]
                    + self.Bk[k] @ self.dnu[k]
                    + self.Bkp[k] @ self.dnu[k + 1]
                    + cp.multiply(self.Sk[k], self.dt[k])
                    + (self.vb_dyn_p[k] - self.vb_dyn_m[k])
                )
                C.append(self.dz[k + 1] + self.z_ref[k + 1] - self.z_m[k + 1] == rhs)

                if self.buff_dyn != "term":
                    C.append(self.vb_dyn_p[k, index_map.z["state"]] >= 0)
                    C.append(self.vb_dyn_m[k, index_map.z["state"]] >= 0)

                if self.ctcs != "term" and n_ctcs > 0:
                    C.append(self.vb_dyn_p[k, index_map.z["ctcs"]] >= 0)
                    C.append(self.vb_dyn_m[k, index_map.z["ctcs"]] >= 0)
                
                # CTCS coupling on extra components
                if method.flags["ctcs"] != "none" and n_ctcs>0:
                    C.append(
                        self.z_ref[k + 1, n:nz] + self.dz[k + 1, n:nz]
                        - (self.z_ref[k, n:nz] + self.dz[k, n:nz]) <= self.eps_ctcs
                    )

                # Free-final-time bounds
                if self.free_T:
                    C.append(self.dt_ref[k] + self.dt[k] <= self.dt_max)
                    C.append(self.dt_ref[k] + self.dt[k] >= self.dt_min)
                    C.append(cp.abs(self.dt[k]) <= self.ddt_max)

                # Control slew (udot)
                for constraint in problem.constraints.get("nodal", "control_rate_limit"):
                    udot_max = constraint.udot_max
                    M_sel = constraint.M_select
                    C.append(
                        M_sel @ (self.nu_ref[k + 1] + self.dnu[k + 1] - (self.nu_ref[k] + self.dnu[k]))
                        <= (self.dt_ref[k] + self.dt[k]) * np.concatenate([udot_max, udot_max])
                    )

            # State box constraints
            for constraint in problem.constraints.get("nodal", "box"):
                x_min_idx = constraint.x_min_idx
                x_max_idx = constraint.x_max_idx
                x_min = constraint.x_min
                x_max = constraint.x_max

                M_select = constraint.M_select

                if constraint.set == "state":

                    C.append(M_select @ (self.z_ref[k, :n] + self.dz[k, :n]) <= np.concatenate([-x_min, x_max]))
                elif constraint.set == "control":
                    C.append(M_select @ (self.nu_ref[k] + self.dnu[k]) <= np.concatenate([-x_min, x_max]))

            # Linearized inequality constraints (path + nfz + custom)
            # if problem.constraints.has("nodal", "nonconvex_inequality","POLYTOPE_OUT","SOC_OUT"):  ### DAN: UPDATE BELOW LINE TO
            if problem.constraints.has("nodal", "nonconvex_inequality"):
                C.append(self.dgdz[k] @ self.dz[k, :n] + self.dgdnu[k] @ self.dnu[k, :m] + self.g0[k] - self.vb_ineq[k] <= 0)
                if str(self.flag_autotune) in {"1", "3", "al-scvx"} and self.vb_ineq[k]:
                    C.append(self.vb_ineq[k] >= 0)

            # convex constraints
            for constraint in problem.constraints.get("nodal", "axis_angle_cone"):
                z = self.z_ref[k] + self.dz[k]
                nu = self.nu_ref[k] + self.dnu[k]
                
                x_idx = constraint.x_idx
                axis = constraint.axis

                if constraint.set == "state":
                    C.append(constraint.cos_theta_max * cp.norm(z[x_idx]) <= axis @ z[x_idx])
                elif constraint.set == "control":
                    C.append(constraint.cos_theta_max * cp.norm(nu[x_idx]) <= axis @ nu[x_idx])

            for constraint in problem.constraints.get("nodal", "max_norm_cone"):
                z = self.z_ref[k] + self.dz[k]
                nu = self.nu_ref[k] + self.dnu[k]

                x_idx = constraint.x_idx
                max_val = constraint.max_val
                
                if constraint.set == "state":
                    C.append(cp.norm(z[x_idx]) <= max_val)
                elif constraint.set == "control":
                    C.append(cp.norm(nu[x_idx]) <= max_val)

            #############################################################################################
            #### ----------------------------------------------------------------------------------- ####
            #### ----------------------------------- FROM DAN -------------------------------------- ####
            #### -------------------------- GENERALIZED CONSTRAINTS -------------------------------- ####


            #### ----------------- DECIDE TO USE A CONSTRAINT BASED ON TIME STEP ------------------- ####
            def use_constraint_at_time_query(time_steps):
                use_constraint = False;
                if time_steps == 'all': use_constraint = True; ## use if applied to all time steps
                elif k in time_steps: use_constraint = True; ## use if positive index is in time_steps
                elif k-N in time_steps: use_constraint = True; ## use if negative index is in time_steps
                return use_constraint

            ############# ----------------------- CONVEX CONSTRAINTS ------------------- ################
            # for constraint in problem.constraints.get('POLYTOPE_IN'):
            for constraint in problem.constraints.get('nodal','AFFINE'):                
                if constraint.convex == True:
                    if use_constraint_at_time_query(constraint.time_steps):
                        z = self.z_ref[k] + self.dz[k]; nu = self.nu_ref[k] + self.dnu[k]
                        idxx = constraint.idx;
                        AA = constraint.A; bb = constraint.b; 
                        if constraint.set == 'state': C.append(AA @ z[idxx] == bb)
                        elif constraint.set == "control": C.append(AA @ nu[idxx] == bb)

            for constraint in problem.constraints.get('nodal','POLYTOPE'):
                if constraint.convex == True:
                    if use_constraint_at_time_query(constraint.time_steps):
                        z = self.z_ref[k] + self.dz[k]; nu = self.nu_ref[k] + self.dnu[k]
                        idxx = constraint.idx;
                        AA = constraint.A; bb = constraint.b; 
                        if constraint.set == 'state': C.append(AA @ z[idxx] <= bb)
                        elif constraint.set == "control": C.append(AA @ nu[idxx] <= bb)
            for constraint in problem.constraints.get('nodal','SOC'):
                if constraint.convex == True:
                    if use_constraint_at_time_query(constraint.time_steps):
                        z = self.z_ref[k] + self.dz[k]; nu = self.nu_ref[k] + self.dnu[k]
                        idxx = constraint.idx;
                        AA = constraint.A; bb = constraint.b;
                        CC = constraint.C; dd = constraint.d;
                        if constraint.set == 'state': C.append(cp.norm(AA@z[idxx] + bb) <= CC @ z[idxx] + dd)
                        if constraint.set == 'control': C.append(cp.norm(AA@nu[idxx] + bb) <= CC @ nu[idxx] + dd)

            ############# -------------------- NONCONVEX CONSTRAINTS ------------------- ################
            # for constraint in problem.constraints.get('POLYTOPE_OUT','SOC_OUT'):
            #     if use_constraint_query(constraint.time_steps):
            #         if constraint.set == 'state': pass 
            #         if constraint.set == 'control': pass 
            
            #### ----------------------------------------------------------------------------------- ####
            #### ----------------------------------------------------------------------------------- ####
            ############################################################################################# 

            # TODO(carlos): think about where to put this, this is a special constraint
            # but general enough to apply to different 6-dof models using quaternions
            for constraint in problem.constraints.get("nodal", "quaternion_cone"):
                z = self.z_ref[k] + self.dz[k]
                quat_start_idx = constraint.quat_start_idx
                C.append(cp.norm(z[quat_start_idx + 2: quat_start_idx + 4]) <= constraint.rhs)
        
        # Fixed-time tying
        if not self.free_T:
            C.append(self.dt == 0)

        # Equal dt tying
        if self.free_T and self.equal_dt and self.dT is not None:
            one = np.ones((self.N - 1, 1)) / (self.N - 1)
            C.append(self.dt == one * self.dT)

        self.cp_constraints += C

    # ============================================================
    # COST FUNCTION (DCP-safe)
    # ============================================================
    def _build_cost_once(self) -> cp.Expression:
        problem, method = self.problem, self.method
        """Full baseline cost: TRUE + TR + 0.5*VIRTUAL + DUAL; gated via flags & autotune."""

        # === TRUE cost (linearized objective) ===
        TRUE = self.flag_true * (
            cp.sum(cp.multiply(self.w_cost_times_dcostdz, self.dz[:,:self.n]))
          + cp.sum(cp.multiply(self.w_cost_times_dcostdnu, self.dnu))
          + cp.sum(self.w_cost_times_cost0)
        )

        if problem.costs.has("min_time"):
            dt = self.dt_ref + self.dt
            time_cost = cp.sum(dt) / (self.N - 1)
            
            TRUE += time_cost

        if problem.costs.has("terminal_state"):
            cost_obj = problem.costs.get("terminal_state")[0]

            zf = self.z_ref[-1] + self.dz[-1]

            x_idx = cost_obj.x_idx
            TRUE += zf[x_idx]

        # === Trust-region penalties ===
        TR = self.flag_tr * (self.wtr_z * cp.sum_squares(self.dz[:, :self.n]) + self.wtr_u * cp.sum_squares(self.dnu))

        # === Virtual buffer penalties ===
        VB = 0.0
        DUAL = 0.0

        # Quadratic penalties 
        if self.flag_autotune in {"0", "2", "3", "al-scvx"}:
            VB = hp.build_virtual_buffer_cost(self)

        # Dual penalties
        if self.flag_autotune in {"1", "3", "al-scvx"}:
            DUAL = hp.build_dual_buffer_cost(self)

        return TRUE + TR + VB + DUAL

    # ============================================================
    # PARAMETER UPDATES AND SOLVE (UNIFIED HISTORY)
    # ============================================================
    def _set_param(self, param: Optional[cp.Parameter], val: np.ndarray) -> None:
        if param is None:
            return
        arr = np.asarray(val)
        if param is self.z_m and arr.shape != (self.N, self.nz):
            arr = arr.reshape(self.N, self.nz)
        param.value = arr

    def _load_inputs(self) -> Dict[str, Any]:
        last_rec = self.iter_data[-1]
        k_prev = int(last_rec.get("iter_num", 0))

        if all(key in last_rec for key in ("z_opt", "nu_opt", "dt_opt", "t_opt")):
            refs = {
                "z_ref": last_rec["z_opt"],
                "nu_ref": last_rec["nu_opt"],
                "dt_ref": last_rec["dt_opt"],
                "t_ref": last_rec["t_opt"],
            }
            weights = copy.deepcopy(last_rec.get("weights", self.method.weights))
            conv_data = last_rec.get("conv_data", {})
        else:
            refs = {
                "z_ref": last_rec["z_ref"],
                "nu_ref": last_rec["nu_ref"],
                "dt_ref": last_rec["dt_ref"],
                "t_ref": last_rec["t_ref"],
            }
            weights = last_rec.get("weights", self.method.weights)
            conv_data = last_rec.get("conv_data", {})

        next_inputs = {
            "iter_num": k_prev + 1,
            **refs,
            "weights": weights,
            "conv_data": conv_data,
        }
        return next_inputs

    def _load_parameters(self, inputs: Dict[str, Any]) -> float:

        problem, method = self.problem, self.method

        start = time.time()
        Ak, Bk, Bkp, Sk, z_minus = discretize.compute_linsys_discrete(
            inputs["z_ref"], inputs["nu_ref"], inputs["dt_ref"], problem, method
        )
        prop_time_ms = (time.time() - start) * 1000.0

        # compute linearized terminal and running costs
        cost, dcostdz, dcostdnu = discretize.compute_linearized_costs(inputs["t_ref"], inputs["z_ref"], inputs["nu_ref"], problem, method)

        if problem.constraints.has("nodal", "nonconvex_inequality"):
            g, dgdz, dgdnu = discretize.compute_nodal_inequality_constraints(inputs["t_ref"], inputs["z_ref"], inputs["nu_ref"], problem, method)
        else:
            g = None
            dgdz = None
            dgdnu = None

        # Dynamics & references
        self._set_param(self.Ak,   Ak)
        self._set_param(self.Bk,   Bk)
        self._set_param(self.Bkp,  Bkp)
        self._set_param(self.Sk,   Sk)
        self._set_param(self.z_m, z_minus)

        # Weights/duals (ensure shapes, fill any scalars/empties)
        W = inputs["weights"]
        self.w_cost = W.get("w_cost", 1.0)
        self.wtr_z.value  = W.get("wtr_z", 1e-2)
        self.wtr_u.value  = W.get("wtr_u", 1e-2)

        self.w_cost_times_dcostdz.value = self.w_cost * dcostdz[:, 0, :]
        self.w_cost_times_dcostdnu.value = self.w_cost * dcostdnu[:, 0, :]
        self.w_cost_times_cost0.value   = self.w_cost * cost[:, 0, 0]
        self.z_ref.value  = inputs["z_ref"]
        self.nu_ref.value  = inputs["nu_ref"]
        self.dt_ref.value = inputs["dt_ref"].reshape(self.N - 1, 1)

        self.nu_ref_sq.value = np.sum(inputs["nu_ref"] * inputs["nu_ref"], axis=1)

        if dgdz is not None:
            self.dgdz.value = dgdz
            self.dgdnu.value = dgdnu
            self.g0.value   = g
        
        if self.free_T:
            self.dt_min.value  = float(method.dt_min)
            self.dt_max.value  = float(method.dt_max)
            self.ddt_max.value = float(method.ddt_max)

        # TODO(Skye): refactor weight loading to reduce code duplication with autotune        
        W_ineq_arr = tools.ensure_shape(W.get("W_ineq", 0.0), (self.N, max(self.n_ineq, 1)))
        W_term_arr = tools.ensure_shape(W.get("W_term", 0.0), (max(self.n_term_total, 1),))
        W_dyn_arr  = tools.ensure_shape(W.get("W_dyn",  0.0), (self.N - 1, max(self.nz, 1)))
        W_plus_real_arr = tools.ensure_shape(W.get("W_plus_real", 0.0), (max(self.Npm_real, 1), max(self.n_plus_real,  1)))
        W_minus_real_arr = tools.ensure_shape(W.get("W_minus_real",0.0), (max(self.Npm_real, 1), max(self.n_minus_real, 1)))
        W_plus_ctcs_arr = tools.ensure_shape(W.get("W_plus_ctcs", 0.0), (max(self.Npm_ctcs, 1), max(self.n_plus_ctcs,  1)))
        W_minus_ctcs_arr = tools.ensure_shape(W.get("W_minus_ctcs",0.0), (max(self.Npm_ctcs, 1), max(self.n_minus_ctcs, 1)))
        # ------------------------------------------------------------------
        # Assign to CVXPY parameters
        self.W_ineq.value  = W_ineq_arr
        self.W_term.value  = W_term_arr
        self.W_dyn.value   = W_dyn_arr
        self.W_plus_real.value  = W_plus_real_arr
        self.W_minus_real.value = W_minus_real_arr
        self.W_plus_ctcs.value  = W_plus_ctcs_arr
        self.W_minus_ctcs.value = W_minus_ctcs_arr
        # ------------------------------------------------------------------
        # Square-rooted parameters (for quadratic penalties)
        self.W_ineq_sqrt.value  = np.sqrt(W_ineq_arr)
        self.W_term_sqrt.value  = np.sqrt(W_term_arr)
        self.W_dyn_sqrt.value   = np.sqrt(W_dyn_arr)
        self.W_plus_real_sqrt.value  = np.sqrt(W_plus_real_arr)
        self.W_minus_real_sqrt.value = np.sqrt(W_minus_real_arr)
        self.W_plus_ctcs_sqrt.value  = np.sqrt(W_plus_ctcs_arr)
        self.W_minus_ctcs_sqrt.value = np.sqrt(W_minus_ctcs_arr)
        # ------------------------------------------------------------------
        # Rowwise scalar reductions (for DCP-safe convex weighting)
        if self.w_ineq_row is not None:
            self.w_ineq_row.value = np.max(W_ineq_arr, axis=1)
        if self.w_dyn_row is not None:
            self.w_dyn_row.value  = np.max(W_dyn_arr, axis=1)
        # ------------------------------------------------------------------
        # Dual variables (unified inequality structure)
        self.dual_ineq.value  = tools.ensure_shape(W.get("dual_ineq", 0.0), (self.N, max(self.n_ineq, 1)))
        self.dual_dyn.value   = tools.ensure_shape(W.get("dual_dyn",  0.0), (self.N - 1, self.nz))
        self.dual_plus_real.value  = tools.ensure_shape(W.get("dual_plus_real", 0.0), (max(self.Npm_real, 1), max(self.n_plus_real,  1)))
        self.dual_minus_real.value = tools.ensure_shape(W.get("dual_minus_real",0.0), (max(self.Npm_real, 1), max(self.n_minus_real, 1)))
        self.dual_plus_ctcs.value  = tools.ensure_shape(W.get("dual_plus_ctcs", 0.0), (max(self.Npm_ctcs, 1), max(self.n_plus_ctcs,  1)))
        self.dual_minus_ctcs.value = tools.ensure_shape(W.get("dual_minus_ctcs",0.0), (max(self.Npm_ctcs, 1), max(self.n_minus_ctcs, 1)))
        self.dual_term.value  = tools.ensure_shape(W.get("dual_term", 0.0), (max(self.n_term_total,1),))


        # ctcs eps
        self.eps_ctcs.value = float(method.conv["eps_ctcs"])

        # cache for optional debug
        inputs["_linsys_cache"] = (Ak, Bk, Bkp, Sk, z_minus)
        return prop_time_ms

    def solve_iteration(self) -> None:
        """
        Single SCP iteration (no return):
          - builds inputs for the iteration from self.iter_data[-1],
          - loads Parameters and solves,
          - assembles a unified record (inputs used + outputs),
          - appends it to self.iter_data.
        """
        # Build inputs for this iteration (refs/weights/conv_data, iter_num)
        input_for_iter = self._load_inputs()

        # Parameter propagation and linearization
        prop_time_ms = self._load_parameters(input_for_iter)

        # Solve subproblem
        solver_name = self.method.solver_opts.get("solver", "ECOS")
        ignore_dpp = self.method.flags.get("ignore_dpp", False)
        self.subproblem.solve(solver=solver_name, warm_start=True, ignore_dpp=ignore_dpp)  # ignore_dpp=True if desired

        # Create unified record for this iteration and append
        iter_record = self._load_outputs(input_for_iter, prop_time_ms)
        iter_record = convergence.check_convergence_tolerance(self.problem, self.method, iter_record)
        iter_record = self._baseline_autotune(self.problem, self.method, iter_record)
        self.iter_data.append(iter_record)

    # ============================================================
    # OUTPUT PACKING (UNIFIED RECORD)
    # ============================================================
    def _load_outputs(self, input_for_iter: Dict[str, Any], prop_time_ms: float) -> Dict[str, Any]:
        # mission, model, method = self.problem.mission, self.problem.model, self.method
        N, n, m = self.N, self.n, self.m

        dz_val, dnu_val = self.dz.value, self.dnu.value
        dt_val = self.dt.value if isinstance(self.dt, cp.expressions.expression.Expression) else self.dt

        rec = copy.deepcopy(input_for_iter)
        rec["subprob"] = self.subproblem

        if self.subproblem is not None:
            compilation_time = getattr(self.subproblem, "compilation_time", None)
            rec["parse_time"] = float(compilation_time or 0.0) * 1000.0
        else:
            rec["parse_time"] = None
        

        if self.subproblem.solver_stats is not None:
            solve_time = getattr(self.subproblem.solver_stats, "solve_time", None)
            rec["solve_time"] = float(solve_time or 0.0) * 1000.0
        else:
            rec["solve_time"] = None

        # raw solver variables (useful for diagnostics)
        rec["dz_s"] = dz_val
        rec["dnu_s"] = dnu_val
        rec["dt_s"] = dt_val

        # outputs (absolute trajectories)
        rec["z_opt"]  = tools.safe_val(dz_val, rows=N, cols=n) + input_for_iter["z_ref"]
        rec["nu_opt"]  = tools.safe_val(dnu_val, rows=N, cols=m) + input_for_iter["nu_ref"]
        rec["dt_opt"] = tools.safe_val(dt_val).squeeze() + input_for_iter["dt_ref"].squeeze()
        rec["t_opt"]  = np.concatenate(([0], np.cumsum(rec["dt_opt"])))
        rec["T_opt"]  = float(np.sum(rec["dt_opt"]))

        # Discretization model (expose for debug/analysis)
        Ak, Bk, Bkp, Sk, z_minus = input_for_iter.get("_linsys_cache", (None, None, None, None, None))
        rec["z_minus"] = self.z_m.value if z_minus is None else z_minus
        rec["Ak"] = self.Ak.value if Ak is None else Ak
        rec["Bk"] = self.Bk.value if Bk is None else Bk
        rec["Bkp"] = self.Bkp.value if Bkp is None else Bkp
        rec["Sk"]  = self.Sk.value if Sk is None else Sk

        # Path residuals and reference cost
        g, _, _ = discretize.compute_nodal_inequality_constraints(rec["t_opt"], rec["z_opt"], rec["nu_opt"], self.problem, self.method)

        rec["cnst_path"] = g
        rec["cost"]      = discretize.compute_linearized_costs(rec["t_opt"], rec["z_opt"], rec["nu_opt"], self.problem, self.method)[0].sum().item()
 
        # Convergence data (buffers, defects, TR cost, ref cost)
        conv = {}
        conv["vb_ineq"] = tools.get_val(self.vb_ineq,  rows=self.N, cols=self.n_ineq) if self.vb_ineq  is not None else np.zeros((self.N,self.n_ineq))
        conv["vb_term"] = tools.get_val(self.vb_term,  rows=1, cols=self.n_term_total) if self.vb_term  is not None else np.zeros((1, self.n_term_total))
        conv["vb_dyn"]  = tools.get_val(self.vb_dyn_p, rows=self.N-1,  cols=self.n_dyn) - tools.get_val(self.vb_dyn_m, rows=self.N-1, cols=self.n_dyn)
        conv["vb_plus_real"] = tools.get_val(self.vb_plus_real, rows=self.Npm_real, cols=self.n_plus_real) if self.vb_plus_real  is not None else np.zeros((self.Npm_real, self.n_plus_real))
        conv["vb_minus_real"] = tools.get_val(self.vb_minus_real, rows=self.Npm_real, cols=self.n_minus_real) if self.vb_minus_real  is not None else np.zeros((self.Npm_real, self.n_minus_real))
        conv["vb_plus_ctcs"] = tools.get_val(self.vb_plus_ctcs, rows=self.Npm_ctcs, cols=self.n_plus_ctcs) if self.vb_plus_ctcs  is not None else np.zeros((self.Npm_ctcs, self.n_plus_ctcs))
        conv["vb_minus_ctcs"] = tools.get_val(self.vb_minus_ctcs, rows=self.Npm_ctcs, cols=self.n_minus_ctcs) if self.vb_minus_ctcs  is not None else np.zeros((self.Npm_ctcs, self.n_minus_ctcs))

        conv["defect"]  = tools.safe_val(self.dz, rows=N, cols=n) + input_for_iter["z_ref"] - self.z_m.value
        conv["Jtr"]     = ( float(self.wtr_z.value) * np.sum(tools.safe_val(self.dz, rows=N, cols=n)**2)
                          + float(self.wtr_u.value) * np.sum(tools.safe_val(self.dnu, rows=N, cols=m)**2) )
        ref_cost = discretize.compute_linearized_costs(input_for_iter["t_ref"], input_for_iter["z_ref"], input_for_iter["nu_ref"], self.problem, self.method)[0].sum().item()
        conv["cost_ref"] = ref_cost

        rec["conv_data"]  = conv
        rec["prop_time"]  = prop_time_ms

        return rec
    
    # ===========================
    # Baseline autotune wrapper
    # ===========================
    def _baseline_autotune(self, problem, method, rec: Dict[str, Any]) -> Dict[str, Any]:
        flag = method.flags["flag_autotune"]
        if flag == "1":
            rec = hp.autotune1(problem, method, rec)
        elif flag == "2":
            rec = hp.autotune2(problem, method, rec)
        elif flag == "3":
            rec = hp.autotune3(problem, method, rec)
        return rec