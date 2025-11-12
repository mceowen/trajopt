"""
Subproblem module: SCP with DPP updates 
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import importlib
import time
import numpy as np
import cvxpy as cp
import jax.numpy as jnp

# trajopt imports
from trajopt.core.modules.method import discretize as discretize
from trajopt.core.modules.method import convexify as convexify
from trajopt.core.modules.method import hyperparameters as hp
from trajopt.core.modules.method import convergence
from trajopt.utils import tools


# =====================================================================================
# Public API: full Sequential Convex Programming (SCP) loop with DPP reuse
# =====================================================================================
def run_scp(problem):
    """
    Run the full Sequential Convex Programming (SCP) loop for a given problem.

    Subproblem maintains iteration history internally:
      - subprob.iter_data[0]: initialization inputs only
      - subprob.iter_data[k>=1]: inputs used at iter k + outputs from that iter
    """

    # Start full problem convergence timer
    time_start = time.perf_counter()

    # Get or create the compiled Subproblem (DPP)
    subprob: Optional[Subproblem] = getattr(problem.method, "subprob", None)
    if subprob is None:
        subprob = Subproblem(problem)
        problem.method.subprob = subprob

    # START SUBPROBLEM CONSTRUCTION / HEADER
    print("-" * 152)
    print(f"                                              ..:: {problem.mission.name}: PTR with Virtual Buffer ::..")
    print("-" * 152)
    print("  Iteration |  Propagation |   Solve   |    Parse   |  log(dz)  |      log(VB)    |   log(VB)   |  log(VB)    | Solve status |  Time of    |   Cost    ")
    print("            |   time [ms]  | time [ms] |  time [ms] |           |  (path + NFZ)   |  (terminal) |  (dynamics) |              |  Flight [s] |           ")
    print("-" * 152)

    max_iter = int(problem.method.conv["iter_max"])

    for _ in range(max_iter + 1):
        subprob.solve_iteration()  # appends a new unified record for this iteration

        latest = subprob.iter_data[-1]
        display_subprob_status(problem, latest)

        if latest.get("converged", False):
            print("Terminated from convergence criteria!")
            break

    if not subprob.iter_data[-1].get("converged", False):
        print("Terminated from hitting maximum iterations!")

    problem.solution = problem.method.subprob.iter_data[-1]

    # Save off convergence time (full time - parse time)
    problem.solution['t_full'] = time.perf_counter() - time_start

    return problem


# ===================================
# Module flags (gating via Parameters)
# ===================================
@dataclass
class ModuleFlags:
    path:  float = 1.0
    nfz:   float = 1.0
    aux:   float = 1.0
    term:  float = 1.0
    dyn:   float = 1.0
    tr:    float = 1.0
    true:  float = 1.0
    dual:  float = 1.0
    vb:    float = 1.0


# =========================
# Subproblem (build-once)
# =========================
class Subproblem:
    """Reusable convex SCP with full baseline functionality & DPP updates.
    """

    def __init__(self, problem) -> None:
        self.problem = problem
        mission = self.problem.mission
        model   = self.problem.model
        method  = self.problem.method

        # derive canonical sizes for quick reuse
        self.N  = int(method.N)
        self.n  = int(model.n)
        self.m  = int(model.m)
        self.nz = int(model.nz)

        self.flags         = method.flags
        self.free_T        = bool(self.flags["free_final_time"])
        self.equal_dt      = bool(self.flags["equal_dt"])
        self.buff_dyn      = self.flags["buff_dyn"]                  # e.g., "l1", "l2", "term", "quad-1", "quad-2"
        self.flag_autotune = self.flags["flag_autotune"]

        # module sizes
        self.n_path  = int(mission.n_path)
        self.n_nfz   = int(mission.n_nfz)
        self.n_aux   = int(getattr(mission, "n_aux", 0))
        self.n_ineq  = int(mission.n_ineq)
        self.n_term  = int(mission.n_term + mission.n_term_ineq)
        self.n_dyn   = int(getattr(mission, "n_dyn", model.nz))
        self.Npm     = int(getattr(method, "Npm", 0))
        self.n_plus  = int(getattr(method, "n_plus", 0))
        self.n_minus = int(getattr(method, "n_minus", 0))

        # Optional module flags as Parameters (enable gating)
        self.flag_path = method.flags.get("flag_path", 1.0)
        self.flag_nfz  = method.flags.get("flag_nfz", 1.0)
        self.flag_aux  = method.flags.get("flag_aux", 1.0)
        self.flag_term = method.flags.get("flag_term", 1.0)
        self.flag_dyn  = method.flags.get("flag_dyn", 1.0)
        self.flag_tr   = method.flags.get("flag_tr", 1.0)
        self.flag_true = method.flags.get("flag_true", 1.0)
        self.flag_dual = method.flags.get("flag_dual", 1.0)
        self.flag_vb   = method.flags.get("flag_vb", 1.0)

        # Bounds (only create when nonzero-length)
        self._init_bounds(mission)

        # Build the DPP graph once
        self._create_variables()
        self._create_parameters()
        self.constraints: List[cp.constraints.constraint.Constraint] = []
        self._build_constraints_once()
        self.cost_expr = self._build_cost_once()

        # apply custom constraints and cost
        mission.custom_constraints(self)
        mission.custom_cost(self)

        # Compile CVXPY problem once
        self.subproblem = cp.Problem(cp.Minimize(self.cost_expr), self.constraints)

        # --------------------------
        # Initialize unified history
        # --------------------------
        self.iter_data: List[Dict[str, Any]] = [{
            "iter_num": 0,  # init only (no outputs yet)
            "zs_ref": problem.method.zs_init,
            "us_ref": problem.method.us_init,
            "dts_ref": problem.method.dts_init,
            "ts_ref": problem.method.ts_init,
            "conv_data": {
                "vb_ineq": np.zeros((self.N, mission.n_ineq)),
                "vb_dyn":  np.zeros((self.N - 1, self.nz)),
                "vb_term": np.zeros((self.n_term, 1)),
            },
            "weights": problem.method.weights,
        }]

    # -------------------------
    # Helper: init bounds
    # -------------------------
    def _init_bounds(self, mission):
        self.z1     = mission.zi if mission.n_init > 0 else None
        if mission.n_init_ineq > 0:
            self.z1_min = mission.zi_min
            self.z1_max = mission.zi_max
        else:
            self.z1_min = self.z1_max = None

        self.zN     = mission.zf if mission.n_term > 0 else None
        if mission.n_term_ineq > 0:
            self.zN_min = mission.zf_min
            self.zN_max = mission.zf_max
        else:
            self.zN_min = self.zN_max = None

        self.u1 = mission.ui if mission.n_init_ctrl > 0 else None
        self.uN = mission.uf if mission.n_term_ctrl > 0 else None

        self.z_min = mission.z_min if hasattr(mission, "z_min") and len(mission.z_min) > 0 else None
        self.z_max = mission.z_max if hasattr(mission, "z_max") and len(mission.z_max) > 0 else None

        if mission.n_ctrl > 0:
            self.u_min  = mission.u_min
            self.u_max  = mission.u_max
        else:
            self.u_min = self.u_max = None

        if mission.n_udot > 0:
            self.udot_max = mission.udot_max
        else:
            self.udot_max = None

    # ============================================================
    # VARIABLE & PARAMETER CREATION
    # ============================================================
    def _create_variables(self) -> None:
        mission, model, method = self.problem.mission, self.problem.model, self.problem.method
        N, n, m, nz = self.N, self.n, self.m, self.nz

        # Core optimization variables
        self.dz = cp.Variable((N, nz), name="dz")
        self.du = cp.Variable((N, m), name="du")

        # Time variable(s)
        if self.free_T:
            if self.equal_dt:
                self.dT = cp.Variable(name="dT")
                self.dt = (1 / (N - 1)) * self.dT * np.ones((N - 1, 1))
            else:
                self.dT = None
                self.dt = cp.Variable((N - 1, 1), name="dt")
        else:
            self.dT = None
            self.dt = cp.Constant(np.zeros((N - 1, 1)))  # CVXPY constant, safe in constraints


        # Virtual buffers (None if zero-sized)
        self.vb_ineq    = cp.Variable((N, mission.n_ineq), name="vb_ineq")   if mission.n_ineq  > 0 else None 
        self.vb_term    = (
                            cp.Variable(self.n_term, name="vb_term")
                            if method.flags.get("buff_dyn") == "term" and self.n_term > 0
                            else cp.Constant(np.zeros(self.n_term)) if self.n_term > 0
                            else None
                        )


        # Dynamics buffers (zeros if 'term')
        if method.flags.get("buff_dyn") == "term":
            self.vb_dyn_p = np.zeros((N - 1, nz))
            self.vb_dyn_m = np.zeros((N - 1, nz))
        else:
            self.vb_dyn_p = cp.Variable((N - 1, nz), name="vb_dyn_p")
            self.vb_dyn_m = cp.Variable((N - 1, nz), name="vb_dyn_m")

        # Aggregate buffers (optional)
        self.vb_plus  = cp.Variable((self.Npm, self.n_plus),  name="vb_plus")  if self.n_plus  > 0 else None
        self.vb_minus = cp.Variable((self.Npm, self.n_minus), name="vb_minus") if self.n_minus > 0 else None

    def _create_parameters(self) -> None:
        mission, model, method = self.problem.mission, self.problem.model, self.problem.method
        N, n, m, nz = self.N, self.n, self.m, self.nz

        # Linearized dynamics & trajectory references
        self.Ak     = cp.Parameter((N - 1, nz, nz), name="Ak")
        self.Bk     = cp.Parameter((N - 1, nz, m), name="Bk")
        self.Bkp    = cp.Parameter((N - 1, nz, m), name="Bkp")
        self.Sk     = cp.Parameter((N - 1, nz),    name="Sk")
        self.zs_m   = cp.Parameter((N, nz),        name="zs_minus")

        self.w_cost_times_dcostdz = cp.Parameter((N, n), name="dcostdz")
        self.w_cost_times_dcostdu = cp.Parameter((N, m), name="dcostdu")
        self.w_cost_times_cost0   = cp.Parameter(N,      name="cost0")

        self.zs_ref  = cp.Parameter((N, nz),    name="zs_ref")
        self.us_ref  = cp.Parameter((N, m),    name="us_ref")
        self.dts_ref = cp.Parameter((N - 1, 1),name="dts_ref")

        # Path/NFZ/AUX linearized constraints
        if mission.n_ineq > 0:
            self.dgdz = cp.Parameter((N, mission.n_ineq, n), name="dgdz")
            self.dgdu = cp.Parameter((N, mission.n_ineq, m), name="dgdu")
            self.g0   = cp.Parameter((N, mission.n_ineq),    name="g0")
        else:
            self.dgdz = self.dgdu = self.g0 = None

        # time-step scalar bounds
        if self.free_T:
            self.dts_min  = cp.Parameter(nonneg=True, name="dts_min")
            self.dts_max  = cp.Parameter(nonneg=True, name="dts_max")
            self.ddts_max = cp.Parameter(nonneg=True, name="ddts_max")
        else:
            self.dts_min = self.dts_max = self.ddts_max = None

        # Weights & Trust Region weights
        self.w_cost = cp.Parameter(nonneg=True, name="w_cost")
        self.wtr_z  = cp.Parameter(nonneg=True, name="wtr_z")
        self.wtr_u  = cp.Parameter(nonneg=True, name="wtr_u")

        # TODO(Skye): refactor these weight parameters to be less redundant later
        self.W_path  = cp.Parameter((N,  max(self.n_path, 1)),  nonneg=True, name="W_path")
        self.W_nfz   = cp.Parameter((N,  max(self.n_nfz, 1)),   nonneg=True, name="W_nfz")
        self.W_aux   = cp.Parameter((N,  max(self.n_aux, 1)),   nonneg=True, name="W_aux")
        self.W_term  = cp.Parameter((max(self.n_term, 1),),     nonneg=True, name="W_term")
        self.W_dyn   = cp.Parameter((N - 1, max(nz, 1)),        nonneg=True, name="W_dyn")
        self.W_plus  = cp.Parameter((max(self.Npm, 1), max(self.n_plus, 1)),  nonneg=True, name="W_plus")
        self.W_minus = cp.Parameter((max(self.Npm, 1), max(self.n_minus, 1)), nonneg=True, name="W_minus")
        # ----------------------------------------------------------------------------------------------------------
        self.W_path_sqrt  = cp.Parameter((N,  max(self.n_path, 1)),  nonneg=True, name="W_path_sqrt")
        self.W_nfz_sqrt   = cp.Parameter((N,  max(self.n_nfz, 1)),   nonneg=True, name="W_nfz_sqrt")
        self.W_aux_sqrt   = cp.Parameter((N,  max(self.n_aux, 1)),   nonneg=True, name="W_aux_sqrt")
        self.W_term_sqrt  = cp.Parameter((max(self.n_term, 1),),     nonneg=True, name="W_term_sqrt")
        self.W_dyn_sqrt   = cp.Parameter((N - 1, max(nz, 1)),        nonneg=True, name="W_dyn_sqrt")
        self.W_plus_sqrt  = cp.Parameter((max(self.Npm, 1), max(self.n_plus, 1)),  nonneg=True, name="W_plus_sqrt")
        self.W_minus_sqrt = cp.Parameter((max(self.Npm, 1), max(self.n_minus, 1)), nonneg=True, name="W_minus_sqrt")
        # ----------------------------------------------------------------------------------------------------------
        self.w_path_row = cp.Parameter(N,  nonneg=True, name="w_path_row")  if self.n_path > 0 else None
        self.w_nfz_row  = cp.Parameter(N,  nonneg=True, name="w_nfz_row")   if self.n_nfz  > 0 else None
        self.w_aux_row  = cp.Parameter(N,  nonneg=True, name="w_aux_row")   if self.n_aux  > 0 else None
        self.w_dyn_row  = cp.Parameter(N - 1, nonneg=True, name="w_dyn_row")
        # ----------------------------------------------------------------------------------------------------------
        self.W_path.value  = np.ones((N,  max(self.n_path, 1)))
        self.W_nfz.value   = np.ones((N,  max(self.n_nfz, 1)))
        self.W_aux.value   = np.ones((N,  max(self.n_aux, 1)))
        self.W_term.value  = np.ones((max(self.n_term, 1),))
        self.W_dyn.value   = np.ones((N - 1, max(nz, 1)))
        self.W_plus.value  = np.ones((max(self.Npm, 1), max(self.n_plus, 1)))
        self.W_minus.value = np.ones((max(self.Npm, 1), max(self.n_minus, 1)))
        # ----------------------------------------------------------------------------------------------------------
        self.W_path_sqrt.value  = np.sqrt(self.W_path.value)
        self.W_nfz_sqrt.value   = np.sqrt(self.W_nfz.value)
        self.W_aux_sqrt.value   = np.sqrt(self.W_aux.value)
        self.W_term_sqrt.value  = np.sqrt(self.W_term.value)
        self.W_dyn_sqrt.value   = np.sqrt(self.W_dyn.value)
        self.W_plus_sqrt.value  = np.sqrt(self.W_plus.value)
        self.W_minus_sqrt.value = np.sqrt(self.W_minus.value)
        # ----------------------------------------------------------------------------------------------------------
        if self.w_path_row is not None:
            self.w_path_row.value = np.max(self.W_path.value, axis=1)
        if self.w_nfz_row is not None:
            self.w_nfz_row.value  = np.max(self.W_nfz.value, axis=1)
        if self.w_aux_row is not None:
            self.w_aux_row.value  = np.max(self.W_aux.value, axis=1)
        self.w_dyn_row.value = np.max(self.W_dyn.value, axis=1)

        # duals (same ≥1-column pattern)
        self.dual_path  = cp.Parameter((N,  max(self.n_path,1)),  name="dual_path")
        self.dual_nfz   = cp.Parameter((N,  max(self.n_nfz,1)),   name="dual_nfz")
        self.dual_aux   = cp.Parameter((N,  max(self.n_aux,1)),   name="dual_aux")
        self.dual_dyn   = cp.Parameter((N - 1, nz),               name="dual_dyn")
        self.dual_plus  = cp.Parameter((max(self.Npm,1), max(self.n_plus,1)),  name="dual_plus")
        self.dual_minus = cp.Parameter((max(self.Npm,1), max(self.n_minus,1)), name="dual_minus")
        self.dual_term  = cp.Parameter((max(self.n_term,1),),                 name="dual_term")

        # CTCS epsilon (scalar)
        self.eps_ctcs = cp.Parameter(nonneg=True, name="eps_ctcs")

    # ============================================================
    # CONSTRAINTS (build-once)
    # ============================================================
    def _build_constraints_once(self) -> None:
        mission, model, method = self.problem.mission, self.problem.model, self.problem.method
        N, n, m, nz = self.N, self.n, self.m, self.nz
        flags = method.flags

        C: List[cp.Constraint] = []

        # Initial control (optional)
        if mission.flags.get("init_ctrl", False) and mission.n_init_ctrl > 0:
            C.append(self.du[0,mission.ui_idx] + self.us_ref[0, mission.ui_idx] == self.u1)

        # Terminal control (optional)
        if mission.flags.get("final_ctrl", False) and mission.n_term_ctrl > 0:
            C.append(self.du[-1,mission.uf_idx] + self.us_ref[-1, mission.uf_idx] == self.uN)

        # Initial equalities / inequalities
        if mission.n_init > 0 and self.z1 is not None:
            C.append(self.dz[0, mission.zi_idx] + self.zs_ref[0, mission.zi_idx] == self.z1)

        if mission.n_init_ineq > 0 and self.z1_min is not None and self.z1_max is not None:
            M_sel = tools.constraint_index_selector(mission.zi_min_idx, mission.zi_max_idx, n)
            C.append(M_sel @ (self.dz[0, :n] + self.zs_ref[0, :n]) <= cp.hstack([-self.z1_min, self.z1_max]))

        # Terminal equalities / inequalities
        if mission.n_term > 0 and self.zN is not None:
            vbN = self.vb_term[:mission.n_term] if self.vb_term is not None else 0.0
            C.append(self.dz[-1, mission.zf_idx] + self.zs_ref[-1, mission.zf_idx] - vbN == self.zN)

        if mission.n_term_ineq > 0 and self.zN_min is not None and self.zN_max is not None:
            nterm = mission.n_term
            vbNiq = self.vb_term[nterm:nterm + mission.n_term_ineq] if self.vb_term is not None else 0.0
            M_sel = tools.constraint_index_selector(mission.zf_min_idx, mission.zf_max_idx, n)
            C.append(M_sel @ (self.dz[-1, :n] + self.zs_ref[-1, :n]) - vbNiq <= cp.hstack([-self.zN_min, self.zN_max]))

        # Per-stage constraints
        for k in range(N):
            if k < N - 1:
                # Discrete dynamics with dyn buffers
                rhs = (
                    self.Ak[k] @ self.dz[k]
                    + self.Bk[k] @ self.du[k]
                    + self.Bkp[k] @ self.du[k + 1]
                    + cp.multiply(self.Sk[k], self.dt[k])
                    + (self.vb_dyn_p[k] - self.vb_dyn_m[k])
                )
                C.append(self.dz[k + 1] + self.zs_ref[k + 1] - self.zs_m[k + 1] == rhs)

                if self.buff_dyn != "term":
                    C.append(self.vb_dyn_p[k] >= 0)
                    C.append(self.vb_dyn_m[k] >= 0)

                # CTCS coupling on extra components
                if flags["ctcs"] and n < nz:
                    C.append(
                        self.zs_ref[k + 1, n:nz] + self.dz[k + 1, n:nz]
                        - (self.zs_ref[k, n:nz] + self.dz[k, n:nz]) <= self.eps_ctcs
                    )

                # Free-final-time bounds
                if self.free_T:
                    C.append(self.dts_ref[k] + self.dt[k] <= self.dts_max)
                    C.append(self.dts_ref[k] + self.dt[k] >= self.dts_min)
                    C.append(cp.abs(self.dt[k]) <= self.ddts_max)

                # Control slew (udot)
                if mission.n_udot > 0 and self.udot_max is not None and k < N - 2:
                    M_sel = tools.constraint_index_selector(mission.udot_max_idx, mission.udot_max_idx, m)
                    C.append(
                        M_sel @ (self.us_ref[k + 1] + self.du[k + 1] - (self.us_ref[k] + self.du[k]))
                        <= (self.dts_ref[k] + self.dt[k]) * cp.hstack([self.udot_max, self.udot_max])
                    )

            # State box constraints
            if mission.n_state > 0 and self.z_min is not None and self.z_max is not None:
                M_sel = tools.constraint_index_selector(mission.z_min_idx, mission.z_max_idx, n)
                C.append(
                    M_sel @ (self.zs_ref[k, :n] + self.dz[k, :n])
                    <= cp.hstack([-self.z_min, self.z_max])
                )

            # Control box constraints
            if mission.n_ctrl > 0 and self.u_min is not None and self.u_max is not None:
                M_sel = tools.constraint_index_selector(mission.u_min_idx, mission.u_max_idx, m)
                C.append(
                    M_sel @ (self.us_ref[k] + self.du[k])
                    <= cp.hstack([-self.u_min, self.u_max])
                )

            # Linearized inequality constraints (path + nfz + aux)
            if mission.n_ineq > 0 and not flags["ctcs"] and self.dgdz is not None:
        
                C.append(self.dgdz[k] @ self.dz[k] + self.dgdu[k] @ self.du[k] + self.g0[k] - self.vb_ineq[k] <= 0)
                if str(self.flag_autotune) in {"1", "3", "al-scvx"} and self.vb_ineq[k]:
                    C.append(self.vb_ineq[k] >= 0)

        # Fixed-time tying
        if not self.free_T:
            C.append(self.dt == 0)

        # Equal dt tying
        if self.free_T and self.equal_dt and self.dT is not None:
            one = np.ones((self.N - 1, 1)) / (self.N - 1)
            C.append(self.dt == one * self.dT)

        self.constraints += C

    # ============================================================
    # COST FUNCTION (DCP-safe)
    # ============================================================
    def _build_cost_once(self) -> cp.Expression:
        mission, model, method = self.problem.mission, self.problem.model, self.problem.method
        """Full baseline cost: TRUE + TR + 0.5*VIRTUAL + DUAL; gated via flags & autotune."""

        # === TRUE cost (linearized objective) ===
        TRUE = self.flag_true * (
            cp.sum(cp.multiply(self.w_cost_times_dcostdz, self.dz[:,:model.n]))
          + cp.sum(cp.multiply(self.w_cost_times_dcostdu, self.du))
          + cp.sum(self.w_cost_times_cost0)
        )

        # === Trust-region penalties ===
        TR = self.flag_tr * (self.wtr_z * cp.sum_squares(self.dz[:, :model.n]) + self.wtr_u * cp.sum_squares(self.du))

        # === Virtual-buffer quadratic penalties ===
        VB = 0.0
        if self.flag_autotune in {"0", "2", "3", "al-scvx"}:

            # Terminal term: weighted quadratic
            if self.vb_term is not None and self.n_term > 0:
                VB += cp.sum_squares(cp.diag(self.W_term_sqrt) @ self.vb_term)

            # Path / NFZ / AUX (loop over time steps)
            if self.vb_ineq is not None and self.n_path > 0:
                for k in range(self.N):
                    # OLD FOR REFERENCE(FEEL FREE TO DELETE)
                    # VB += cp.sum_squares(cp.diag(self.W_path_sqrt[k, :]) @ self.vb_path[k])
                    
                    VB += cp.sum_squares(cp.diag(self.W_path_sqrt[k, :]) @ self.vb_ineq[k, mission.path_idx])
            
            if self.vb_ineq is not None and self.n_nfz > 0:
                for k in range(self.N):
                    # OLD FOR REFERENCE(FEEL FREE TO DELETE)
                    # VB += cp.sum_squares(cp.diag(self.W_nfz_sqrt[k, :]) @ self.vb_nfz[k])
                    
                    VB += cp.sum_squares(cp.diag(self.W_nfz_sqrt[k, :]) @ self.vb_ineq[k, mission.nfz_idx])
            
            if self.vb_ineq is not None and self.n_aux > 0:
                for k in range(self.N):
                    # OLD FOR REFERENCE(FEEL FREE TO DELETE)
                    # VB += cp.sum_squares(cp.diag(self.W_aux_sqrt[k, :]) @ self.vb_aux[k])
                    
                    VB += cp.sum_squares(cp.diag(self.W_nfz_sqrt[k, :]) @ self.vb_ineq[k, mission.aux_idx])

            # Dynamics (L1 / L2 / quadratic penalties)
            diff = self.vb_dyn_p - self.vb_dyn_m 
            if self.buff_dyn == "l1":
                for k in range(self.N - 1):
                    VB += self.w_dyn_row[k] * cp.norm1(diff[k])

            elif self.buff_dyn == "l2":
                for k in range(self.N - 1):
                    VB += cp.sum_squares(cp.diag(self.W_dyn_sqrt[k, :]) @ diff[k])

            elif self.buff_dyn in {"quad-1", "quad-2"}:
                if self.vb_plus is not None and self.n_plus > 0:
                    for k in range(max(self.Npm, 1)):
                        VB += cp.sum_squares(cp.diag(self.W_plus_sqrt[k, :]) @ self.vb_plus[k])
                if self.vb_minus is not None and self.n_minus > 0:
                    for k in range(max(self.Npm, 1)):
                        VB += cp.sum_squares(cp.diag(self.W_minus_sqrt[k, :]) @ self.vb_minus[k])


        VB = 0.5 * self.flag_vb * VB

        # === Dual costs ===
        DUAL = 0.0
        if self.flag_autotune in {"1", "3", "al-scvx"}:

            # OLD FOR REFERENCE(FEEL FREE TO DELETE)
            # if self.vb_path  is not None and self.n_path  > 0: DUAL += cp.sum(cp.multiply(self.vb_path,  self.dual_path))
            # if self.vb_nfz   is not None and self.n_nfz   > 0: DUAL += cp.sum(cp.multiply(self.vb_nfz,   self.dual_nfz))
            # if self.vb_aux   is not None and self.n_aux   > 0: DUAL += cp.sum(cp.multiply(self.vb_aux,   self.dual_aux))

            if self.vb_ineq  is not None and self.n_path  > 0: DUAL += cp.sum(cp.multiply(self.vb_ineq[:, mission.path_idx],  self.dual_path))
            if self.vb_ineq  is not None and self.n_nfz   > 0: DUAL += cp.sum(cp.multiply(self.vb_ineq[:, mission.nfz_idx],   self.dual_nfz))
            if self.vb_ineq  is not None and self.n_aux   > 0: DUAL += cp.sum(cp.multiply(self.vb_ineq[:, mission.aux_idx],   self.dual_aux))

            diff = self.vb_dyn_p - self.vb_dyn_m
            DUAL += cp.sum(cp.multiply(diff, self.dual_dyn))
            if self.vb_plus  is not None and self.n_plus  > 0: DUAL += cp.sum(cp.multiply(self.vb_plus,  self.dual_plus))
            if self.vb_minus is not None and self.n_minus > 0: DUAL += cp.sum(cp.multiply(self.vb_minus, self.dual_minus))
            if self.vb_term  is not None and self.n_term  > 0: DUAL += self.dual_term @ self.vb_term

        return TRUE + TR + VB + DUAL

    # ============================================================
    # PARAMETER UPDATES AND SOLVE (UNIFIED HISTORY)
    # ============================================================
    def _set_param(self, param: Optional[cp.Parameter], val: np.ndarray) -> None:
        if param is None:
            return
        arr = np.asarray(val)
        if param is self.zs_m and arr.shape != (self.N, self.nz):
            arr = arr.reshape(self.N, self.nz)
        param.value = arr

    def _load_inputs(self) -> Dict[str, Any]:
        last_rec = self.iter_data[-1]
        k_prev = int(last_rec.get("iter_num", 0))

        if all(key in last_rec for key in ("zs", "us", "dts", "ts")):
            refs = {
                "zs_ref": last_rec["zs"],
                "us_ref": last_rec["us"],
                "dts_ref": last_rec["dts"],
                "ts_ref": last_rec["ts"],
            }
            weights = last_rec.get("weights", self.problem.method.weights)
            conv_data = last_rec.get("conv_data", {})
        else:
            refs = {
                "zs_ref": last_rec["zs_ref"],
                "us_ref": last_rec["us_ref"],
                "dts_ref": last_rec["dts_ref"],
                "ts_ref": last_rec["ts_ref"],
            }
            weights = last_rec.get("weights", self.problem.method.weights)
            conv_data = last_rec.get("conv_data", {})

        next_inputs = {
            "iter_num": k_prev + 1,
            **refs,
            "weights": weights,
            "conv_data": conv_data,
        }
        return next_inputs

    def _load_parameters(self, inputs: Dict[str, Any]) -> float:
        mission, model, method = self.problem.mission, self.problem.model, self.problem.method

        start = time.time()
        Ak, Bk, Bkp, Sk, zs_minus = discretize.compute_linsys_discrete(
            inputs["zs_ref"], inputs["us_ref"], inputs["dts_ref"], self.problem
        )
        prop_time_ms = (time.time() - start) * 1000.0

        # compute linearized terminal and running costs
        cost, dcostdz, dcostdu = discretize.compute_linearized_costs(inputs["ts_ref"], inputs["zs_ref"], inputs["us_ref"], self.problem)
        g, dgdz, dgdu = discretize.compute_nodal_inequality_constraints(inputs["ts_ref"], inputs["zs_ref"], inputs["us_ref"], self.problem)

        # Dynamics & references
        self._set_param(self.Ak,   Ak)
        self._set_param(self.Bk,   Bk)
        self._set_param(self.Bkp,  Bkp)
        self._set_param(self.Sk,   Sk)
        self._set_param(self.zs_m, zs_minus)

        # Weights/duals (ensure shapes, fill any scalars/empties)
        W = inputs["weights"]
        self.w_cost = W.get("w_cost", 1.0)
        self.wtr_z.value  = W.get("wtr_z", 1e-2)
        self.wtr_u.value  = W.get("wtr_u", 1e-2)

        self.w_cost_times_dcostdz.value = self.w_cost * dcostdz[:, 0, :]
        self.w_cost_times_dcostdu.value = self.w_cost * dcostdu[:, 0, :]
        self.w_cost_times_cost0.value   = self.w_cost * cost[:, 0, 0]
        self.zs_ref.value  = inputs["zs_ref"]
        self.us_ref.value  = inputs["us_ref"]
        self.dts_ref.value = inputs["dts_ref"].reshape(self.N - 1, 1)

        if dgdz is not None:
            self.dgdz.value = dgdz
            self.dgdu.value = dgdu
            self.g0.value   = g
        
        if self.free_T:
            self.dts_min.value  = float(method.dts_min)
            self.dts_max.value  = float(method.dts_max)
            self.ddts_max.value = float(method.ddts_max)

        # TODO(Skye): refactor weight loading to reduce code duplication with autotune
        W_path_arr  = tools.ensure_shape(W.get("W_path",  0.0), (self.N,  max(self.n_path, 1)))
        W_nfz_arr   = tools.ensure_shape(W.get("W_nfz",   0.0), (self.N,  max(self.n_nfz,  1)))
        W_aux_arr   = tools.ensure_shape(W.get("W_aux",   0.0), (self.N,  max(self.n_aux,  1)))
        W_term_arr  = tools.ensure_shape(W.get("W_term",  0.0), (max(self.n_term, 1),))
        W_dyn_arr   = tools.ensure_shape(W.get("W_dyn",   0.0), (self.N - 1, max(self.nz, 1)))
        W_plus_arr  = tools.ensure_shape(W.get("W_plus",  0.0), (max(self.Npm, 1), max(self.n_plus,  1)))
        W_minus_arr = tools.ensure_shape(W.get("W_minus", 0.0), (max(self.Npm, 1), max(self.n_minus, 1)))
        # ------------------------------------------------------------------
        self.W_path.value  = W_path_arr
        self.W_nfz.value   = W_nfz_arr
        self.W_aux.value   = W_aux_arr
        self.W_term.value  = W_term_arr
        self.W_dyn.value   = W_dyn_arr
        self.W_plus.value  = W_plus_arr
        self.W_minus.value = W_minus_arr
        # ------------------------------------------------------------------
        self.W_path_sqrt.value  = np.sqrt(W_path_arr)
        self.W_nfz_sqrt.value   = np.sqrt(W_nfz_arr)
        self.W_aux_sqrt.value   = np.sqrt(W_aux_arr)
        self.W_term_sqrt.value  = np.sqrt(W_term_arr)
        self.W_dyn_sqrt.value   = np.sqrt(W_dyn_arr)
        self.W_plus_sqrt.value  = np.sqrt(W_plus_arr)
        self.W_minus_sqrt.value = np.sqrt(W_minus_arr)
        # ------------------------------------------------------------------
        if self.w_path_row is not None:
            self.w_path_row.value = np.max(W_path_arr, axis=1)
        if self.w_nfz_row is not None:
            self.w_nfz_row.value = np.max(W_nfz_arr, axis=1)
        if self.w_aux_row is not None:
            self.w_aux_row.value = np.max(W_aux_arr, axis=1)
        if self.w_dyn_row is not None:
            self.w_dyn_row.value = np.max(W_dyn_arr, axis=1)

        # duals
        self.dual_path.value  = tools.ensure_shape(W.get("dual_path", 0.0), (self.N,  max(self.n_path, 1)))
        self.dual_nfz.value   = tools.ensure_shape(W.get("dual_nfz",  0.0), (self.N,  max(self.n_nfz,  1)))
        self.dual_aux.value   = tools.ensure_shape(W.get("dual_aux",  0.0), (self.N,  max(self.n_aux,  1)))
        self.dual_dyn.value   = tools.ensure_shape(W.get("dual_dyn",  0.0), (self.N - 1, self.nz))
        self.dual_plus.value  = tools.ensure_shape(W.get("dual_plus", 0.0), (max(self.Npm, 1), max(self.n_plus,  1)))
        self.dual_minus.value = tools.ensure_shape(W.get("dual_minus",0.0), (max(self.Npm, 1), max(self.n_minus, 1)))
        self.dual_term.value  = tools.ensure_shape(W.get("dual_term", 0.0), (max(self.n_term,1),))

        # ctcs eps
        self.eps_ctcs.value = float(self.problem.method.conv["eps_ctcs"])

        # cache for optional debug
        inputs["_linsys_cache"] = (Ak, Bk, Bkp, Sk, zs_minus)
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
        inputs_for_iter = self._load_inputs()

        # Parameter propagation and linearization
        prop_time_ms = self._load_parameters(inputs_for_iter)

        # Solve subproblem
        solver_name = self.problem.method.solver_opts.get("solver", "ECOS")
        self.subproblem.solve(solver=solver_name, warm_start=True)  # ignore_dpp=True if desired

        # Create unified record for this iteration and append
        iter_record = self._load_outputs(inputs_for_iter, prop_time_ms)
        iter_record = convergence.check_convergence_tolerance(self.problem, self, iter_record)
        iter_record = baseline_autotune(self.problem, {}, iter_record)
        self.iter_data.append(iter_record)

    # ============================================================
    # OUTPUT PACKING (UNIFIED RECORD)
    # ============================================================
    def _load_outputs(self, inputs_for_iter: Dict[str, Any], prop_time_ms: float) -> Dict[str, Any]:
        mission, model, method = self.problem.mission, self.problem.model, self.problem.method
        N, n, m = self.N, self.n, self.m

        dz_val, du_val = self.dz.value, self.du.value
        dt_val = self.dt.value if isinstance(self.dt, cp.expressions.expression.Expression) else self.dt

        rec: Dict[str, Any] = dict(inputs_for_iter)  # include exact inputs used this iteration
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
        rec["du_s"] = du_val
        rec["dt_s"] = dt_val

        # outputs (absolute trajectories)
        rec["zs"]  = tools.safe_val(dz_val, rows=N, cols=n) + inputs_for_iter["zs_ref"]
        rec["us"]  = tools.safe_val(du_val, rows=N, cols=m) + inputs_for_iter["us_ref"]
        rec["dts"] = tools.safe_val(dt_val).squeeze() + inputs_for_iter["dts_ref"].squeeze()
        rec["ts"]  = np.concatenate(([0], np.cumsum(rec["dts"])))
        rec["Ts"]  = float(np.sum(rec["dts"]))

        # Discretization model (expose for debug/analysis)
        Ak, Bk, Bkp, Sk, zs_minus = inputs_for_iter.get("_linsys_cache", (None, None, None, None, None))
        rec["zs_minus"] = self.zs_m.value if zs_minus is None else zs_minus
        rec["Ak"] = self.Ak.value if Ak is None else Ak
        rec["Bk"] = self.Bk.value if Bk is None else Bk
        rec["Bkp"] = self.Bkp.value if Bkp is None else Bkp
        rec["Sk"]  = self.Sk.value if Sk is None else Sk

        # Path residuals and reference cost
        g, _, _ = discretize.compute_nodal_inequality_constraints(rec["ts"], rec["zs"], rec["us"], self.problem)

        rec["cnst_path"] = g
        rec["cost"]      = discretize.compute_linearized_costs(rec["ts"], rec["zs"], rec["us"], self.problem)[0].sum().item()
 
        # Convergence data (buffers, defects, TR cost, ref cost)
        conv = {}
        conv["soln"]    = self.subproblem
        conv["vb_ineq"] = tools.get_val(self.vb_ineq,  rows=self.n_ineq, cols=self.N) if self.vb_ineq  is not None else np.zeros((self.n_ineq,  self.N))
        conv["vb_term"] = tools.get_val(self.vb_term,  rows=self.n_term, cols=1)      if self.vb_term  is not None else np.zeros((self.n_term,  1))
        conv["vb_dyn"]  = tools.get_val(self.vb_dyn_p, rows=self.n_dyn,  cols=self.N-1) - tools.get_val(self.vb_dyn_m, rows=self.n_dyn, cols=self.N-1)

        conv["defect"]  = tools.safe_val(self.dz, rows=N, cols=n) + inputs_for_iter["zs_ref"] - self.zs_m.value
        conv["Jtr"]     = ( float(self.wtr_z.value) * np.sum(tools.safe_val(self.dz, rows=N, cols=n)**2)
                          + float(self.wtr_u.value) * np.sum(tools.safe_val(self.du, rows=N, cols=m)**2) )
        ref_cost = discretize.compute_linearized_costs(inputs_for_iter["ts_ref"], inputs_for_iter["zs_ref"], inputs_for_iter["us_ref"], self.problem)[0].sum().item()
        conv["cost_ref"] = ref_cost

        rec["conv_data"]  = conv
        rec["prop_time"]  = prop_time_ms

        return rec


# ===========================
# Baseline autotune wrapper
# ===========================
def baseline_autotune(problem, _unused, rec: Dict[str, Any]) -> Dict[str, Any]:
    flag = problem.method.flags["flag_autotune"]
    if flag == 1:
        rec = hp.autotune1(problem, {}, rec)
    elif flag == 2:
        rec = hp.autotune2(problem, {}, rec)
    elif flag == 3:
        rec = hp.autotune3(problem, {}, rec)
    return rec


# ==========================================
# Iteration status printout (unified record)
# ==========================================
def display_subprob_status(problem, rec: Dict[str, Any]) -> None:
    conv = rec.get("conv_data", {})

    chk_feas_path = conv.get("chk_feas_path", 0.0)
    chk_feas_nfz  = conv.get("chk_feas_nfz", 0.0)
    ineq_vb       = chk_feas_path + (chk_feas_nfz if chk_feas_nfz != 0 else 0.0)

    chk_dz        = conv.get("chk_dz", 1e-12)
    chk_feas_term = conv.get("chk_feas_term", 1e-12)
    chk_feas_dyn  = conv.get("chk_feas_dyn", 1e-12)

    log_dz      = np.log10(max(chk_dz, 1e-12))
    log_vb_ineq = np.log10(max(ineq_vb, 1e-12))
    log_vb_term = np.log10(max(chk_feas_term, 1e-12))
    log_vb_dyn  = np.log10(max(chk_feas_dyn, 1e-12))

    solve_stat  = conv.get("status", "UNKNOWN")
    iter_num    = int(rec.get("iter_num", -1))

    nt    = float(problem.method.nondim.get("nt", 1.0))
    ncost = float(problem.method.nondim.get("ncost", 1.0))

    Ts   = float(rec.get("Ts", 0.0))
    cost = float(rec.get("cost", 0.0))

    prop_ms = float(rec.get("prop_time", 0.0) or 0.0)
    solve_ms= float(rec.get("solve_time", 0.0) or 0.0)
    parse_ms= float(rec.get("parse_time", 0.0) or 0.0)

    print(
        "     {:02d}     |    {:07.1f}   |   {:06.1f}  |   {:06.1f}   |   {:+04.1f}    |      {:+05.1f}      |    {:+05.1f}    |     {:+05.1f}   |    {:s}    |   {:4.2f}   |  {:4.1f}".format(
            iter_num,
            prop_ms,
            solve_ms,
            parse_ms,
            log_dz,
            log_vb_ineq,
            log_vb_term,
            log_vb_dyn,
            str(solve_stat),
            Ts * nt,
            cost * ncost
        )
    )
