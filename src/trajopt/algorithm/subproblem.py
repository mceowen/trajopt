"""
Subproblem module: SCP subproblem with DPP updates.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import importlib
import time
import numpy as np
import cvxpy as cp

# trajopt imports
from trajopt.algorithm import discretization as discretize
from trajopt.algorithm import convexification as convexify
from trajopt.algorithm import hyperparameters as hp
from trajopt.algorithm import convergence
from trajopt.utils import tools


# =====================================================================================
# Public API (single-entry): caches/reuses one compiled Subproblem per `problem`
# =====================================================================================

def solve_subproblem(problem) -> Dict[str, Any]:
    if not hasattr(problem, "algorithm"):
        problem.algorithm = {}

    subprob: Optional[Subproblem] = problem.algorithm.get("subprob", None)
    if subprob is None:
        subprob = Subproblem(problem)
        problem.algorithm["subprob"] = subprob

    return subprob.solve_iteration()


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
    """Reusable convex subproblem with full baseline functionality & DPP updates."""

    def __init__(self, problem) -> None:
        self.problem = problem

        # derive canonical sizes for quick reuse
        self.N  = int(self.problem.method.N)
        self.n  = int(self.problem.model.n)
        self.m  = int(self.problem.model.m)
        self.nz = int(self.problem.model.nz)

        self.bools         = self.problem.method.bools
        self.free_T        = bool(self.bools["free_final_time"])
        self.equal_dt      = bool(self.bools["equal_dt"])
        self.buff_dyn      = self.bools["buff_dyn"]
        self.flag_autotune = self.bools["flag_autotune"]

        # module sizes
        self.n_path  = int(self.problem.mission.n_path)
        self.n_nfz   = int(self.problem.mission.n_nfz)
        self.n_aux   = int(getattr(self.problem.mission, "n_aux", 0))
        self.n_term  = int(self.problem.mission.n_term + self.problem.mission.n_term_ineq)
        self.n_dyn   = int(getattr(self.problem.mission, "n_dyn", self.problem.model.nz))
        self.Npm     = int(getattr(self.problem.method, "Npm", 0))
        self.n_plus  = int(getattr(self.problem.method, "n_plus", 0))
        self.n_minus = int(getattr(self.problem.method, "n_minus", 0))

        # Build the DPP graph once
        self._create_variables()
        self._create_parameters()
        self.constraints: List[cp.constraints.constraint.Constraint] = []
        self._build_constraints_once()
        self.cost_expr = self._build_cost_once()

        # Optional custom hooks from model package
        self.invoke_custom_functions()

        # Compile CVXPY problem once
        self.subproblem = cp.Problem(cp.Minimize(self.cost_expr), self.constraints)

    # ============================================================
    # VARIABLE & PARAMETER CREATION
    # ============================================================
    def _create_variables(self) -> None:
        mission, model, method = self.problem.mission, self.problem.model, self.problem.method
        N, n, m, nz = self.N, self.n, self.m, self.nz

        # Core optimization variables
        self.dz = cp.Variable((N, n), name="dz")
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
            self.dT, self.dt = None, np.zeros((N - 1, 1))  # constant, not a Variable

        # Virtual buffers (None if zero-sized)
        self.vb_path  = cp.Variable((N,  mission.n_path), name="vb_path")  if mission.n_path  > 0 else None
        self.vb_nfz   = cp.Variable((N,  mission.n_nfz),  name="vb_nfz")   if mission.n_nfz   > 0 else None
        self.vb_aux   = cp.Variable((N,  self.n_aux),     name="vb_aux")   if self.n_aux      > 0 else None
        self.vb_term  = cp.Variable(self.n_term,          name="vb_term")  if self.n_term     > 0 else None

        # Dynamics buffers (zeros if 'term')
        if method.bools.get("buff_dyn") == "term":
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
        self.Ak     = cp.Parameter((N - 1, n, n), name="Ak")
        self.Bk     = cp.Parameter((N - 1, n, m), name="Bk")
        self.Bkp    = cp.Parameter((N - 1, n, m), name="Bkp")
        self.Sk     = cp.Parameter((N - 1, n),    name="Sk")
        self.zs_m   = cp.Parameter((N, n),        name="zs_minus")

        self.dcostdz = cp.Parameter((N, n), name="dcostdz")
        self.dcostdu = cp.Parameter((N, m), name="dcostdu")
        self.cost0   = cp.Parameter(N,      name="cost0")

        self.zs_ref  = cp.Parameter((N, n),    name="zs_ref")
        self.us_ref  = cp.Parameter((N, m),    name="us_ref")
        self.dts_ref = cp.Parameter((N - 1, 1),name="dts_ref")

        # Path/NFZ/AUX linearized constraints
        n_ineq_cols = self.n_path + self.n_nfz + self.n_aux
        if n_ineq_cols > 0:
            self.dgdz = cp.Parameter((N, n_ineq_cols, n), name="dgdz")
            self.dgdu = cp.Parameter((N, n_ineq_cols, m), name="dgdu")
            self.g0   = cp.Parameter((N, n_ineq_cols),    name="g0")
        else:
            self.dgdz = self.dgdu = self.g0 = None

        # Bounds (only create when nonzero-length)
        self.z1     = cp.Parameter(mission.n_init, name="z1") if mission.n_init > 0 else None
        if mission.n_init_ineq > 0:
            self.z1_min = cp.Parameter(mission.n_init_ineq, name="z1_min")
            self.z1_max = cp.Parameter(mission.n_init_ineq, name="z1_max")
        else:
            self.z1_min = self.z1_max = None

        self.zN     = cp.Parameter(mission.n_term, name="zN") if mission.n_term > 0 else None
        if mission.n_term_ineq > 0:
            self.zN_min = cp.Parameter(mission.n_term_ineq, name="zN_min")
            self.zN_max = cp.Parameter(mission.n_term_ineq, name="zN_max")
        else:
            self.zN_min = self.zN_max = None

        self.z_min = cp.Parameter(len(getattr(mission, "z_min", [])), name="z_min") if hasattr(mission, "z_min") and len(mission.z_min) > 0 else None
        self.z_max = cp.Parameter(len(getattr(mission, "z_max", [])), name="z_max") if hasattr(mission, "z_max") and len(mission.z_max) > 0 else None

        if mission.n_ctrl > 0:
            self.u_min  = cp.Parameter(mission.n_ctrl, name="u_min")
            self.u_max  = cp.Parameter(mission.n_ctrl, name="u_max")
        else:
            self.u_min = self.u_max = None

        if mission.n_udot > 0:
            self.udot_max = cp.Parameter(mission.n_udot, name="udot_max")
        else:
            self.udot_max = None

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

        # Elementwise weights (kept with at least 1 column for DPP)
        self.W_path  = cp.Parameter((N,  max(self.n_path, 1)),  nonneg=True, name="W_path")
        self.W_nfz   = cp.Parameter((N,  max(self.n_nfz, 1)),   nonneg=True, name="W_nfz")
        self.W_aux   = cp.Parameter((N,  max(self.n_aux, 1)),   nonneg=True, name="W_aux")
        self.W_term  = cp.Parameter((max(self.n_term, 1),),     nonneg=True, name="W_term")
        self.W_dyn   = cp.Parameter((N - 1, max(nz, 1)),        nonneg=True, name="W_dyn")
        self.W_plus  = cp.Parameter((max(self.Npm, 1), max(self.n_plus, 1)),  nonneg=True, name="W_plus")
        self.W_minus = cp.Parameter((max(self.Npm, 1), max(self.n_minus, 1)), nonneg=True, name="W_minus")

        # Row weights (DCP-safe convex weighting of row norms)
        self.w_path_row = cp.Parameter(N,  nonneg=True, name="w_path_row")  if self.n_path > 0 else None
        self.w_nfz_row  = cp.Parameter(N,  nonneg=True, name="w_nfz_row")   if self.n_nfz  > 0 else None
        self.w_aux_row  = cp.Parameter(N,  nonneg=True, name="w_aux_row")   if self.n_aux  > 0 else None
        self.w_dyn_row  = cp.Parameter(N-1,nonneg=True, name="w_dyn_row")

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

        # Optional module flags as Parameters (enable gating)
        self.flag_path = cp.Parameter(nonneg=True, value=1.0, name="flag_path")
        self.flag_nfz  = cp.Parameter(nonneg=True, value=1.0, name="flag_nfz")
        self.flag_aux  = cp.Parameter(nonneg=True, value=1.0, name="flag_aux")
        self.flag_term = cp.Parameter(nonneg=True, value=1.0, name="flag_term")
        self.flag_dyn  = cp.Parameter(nonneg=True, value=1.0, name="flag_dyn")
        self.flag_tr   = cp.Parameter(nonneg=True, value=1.0, name="flag_tr")
        self.flag_true = cp.Parameter(nonneg=True, value=1.0, name="flag_true")
        self.flag_dual = cp.Parameter(nonneg=True, value=1.0, name="flag_dual")
        self.flag_vb   = cp.Parameter(nonneg=True, value=1.0, name="flag_vb")

    # ============================================================
    # CONSTRAINTS (build-once)
    # ============================================================
    def _build_constraints_once(self) -> None:
        mission, model, method = self.problem.mission, self.problem.model, self.problem.method
        N, n, m, nz = self.N, self.n, self.m, self.nz
        bools = method.bools

        C: List[cp.Constraint] = []

        # Initial control (optional)
        if method.bools.get("init_ctrl", False) and mission.n_ctrl > 0:
            C.append(self.du[:, 0] == 0)

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
                if bools["ctcs"] and n < nz:
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
            n_ineq_cols = int(mission.n_path + mission.n_nfz + getattr(mission, "n_aux", 0))
            if n_ineq_cols > 0 and not bools["ctcs"] and self.dgdz is not None:
                vb_parts = []
                if self.vb_path is not None: vb_parts.append(self.vb_path[k])
                if self.vb_nfz  is not None: vb_parts.append(self.vb_nfz[k])
                if self.vb_aux  is not None: vb_parts.append(self.vb_aux[k])
                vb_stack = cp.hstack(vb_parts) if vb_parts else 0.0

                C.append(self.dgdz[k] @ self.dz[k] + self.dgdu[k] @ self.du[k] + self.g0[k] - vb_stack <= 0)
                if str(self.flag_autotune) in {"1", "3", "al-scvx"} and vb_parts:
                    C.append(vb_stack >= 0)

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
        """Full baseline cost: TRUE + TR + 0.5*VIRTUAL + DUAL; gated via flags & autotune."""

        # === TRUE cost (linearized objective) ===
        TRUE = self.flag_true * self.w_cost * (
            cp.sum(cp.multiply(self.dcostdz, self.dz))
            + cp.sum(cp.multiply(self.dcostdu, self.du))
            + cp.sum(self.cost0)
        )

        # === Trust-region penalties ===
        TR = self.flag_tr * (
            self.wtr_z * cp.sum_squares(self.dz) + self.wtr_u * cp.sum_squares(self.du)
        )

        # === Virtual-buffer quadratic penalties ===
        VB = 0.0
        if self.flag_autotune in {"0", "2", "3", "al-scvx"}:

            # Terminal term: weighted quadratic
            if self.vb_term is not None and self.n_term > 0:
                VB += cp.quad_form(self.vb_term, cp.diag(self.W_term))

            # Path / NFZ / AUX (loop over time steps)
            if self.vb_path is not None and self.n_path > 0:
                for k in range(self.N):
                    VB += cp.quad_form(self.vb_path[k], cp.diag(self.W_path[k]))
            if self.vb_nfz is not None and self.n_nfz > 0:
                for k in range(self.N):
                    VB += cp.quad_form(self.vb_nfz[k], cp.diag(self.W_nfz[k]))
            if self.vb_aux is not None and self.n_aux > 0:
                for k in range(self.N):
                    VB += cp.quad_form(self.vb_aux[k], cp.diag(self.W_aux[k]))

            # Dynamics (quadratic penalties)
            diff = self.vb_dyn_p - self.vb_dyn_m
            if self.buff_dyn in {"l1", "l2"}:
                for k in range(self.N - 1):
                    VB += cp.quad_form(diff[k], cp.diag(self.W_dyn[k]))
            elif self.buff_dyn in {"quad-1", "quad-2"}:
                if self.vb_plus is not None and self.n_plus > 0:
                    for k in range(max(self.Npm, 1)):
                        VB += cp.quad_form(self.vb_plus[k], cp.diag(self.W_plus[k]))
                if self.vb_minus is not None and self.n_minus > 0:
                    for k in range(max(self.Npm, 1)):
                        VB += cp.quad_form(self.vb_minus[k], cp.diag(self.W_minus[k]))

        VB = 0.5 * self.flag_vb * VB

        # === Dual costs ===
        DUAL = 0.0
        if self.flag_autotune in {"1", "3", "al-scvx"}:
            if self.vb_path  is not None and self.n_path  > 0: DUAL += cp.sum(cp.multiply(self.vb_path,  self.dual_path))
            if self.vb_nfz   is not None and self.n_nfz   > 0: DUAL += cp.sum(cp.multiply(self.vb_nfz,   self.dual_nfz))
            if self.vb_aux   is not None and self.n_aux   > 0: DUAL += cp.sum(cp.multiply(self.vb_aux,   self.dual_aux))
            diff = self.vb_dyn_p - self.vb_dyn_m
            DUAL += cp.sum(cp.multiply(diff, self.dual_dyn))
            if self.vb_plus  is not None and self.n_plus  > 0: DUAL += cp.sum(cp.multiply(self.vb_plus,  self.dual_plus))
            if self.vb_minus is not None and self.n_minus > 0: DUAL += cp.sum(cp.multiply(self.vb_minus, self.dual_minus))
            if self.vb_term  is not None and self.n_term  > 0: DUAL += self.dual_term @ self.vb_term

        return TRUE + TR + VB + DUAL


    # ============================================================
    # CUSTOM MODEL HOOKS
    # ============================================================
    def invoke_custom_functions(self) -> None:
        """Invoke user-defined custom functions in trajopt.problem_models.<model_name>"""
        model_name = getattr(self.problem.method, "model_name", None)
        if not model_name:
            return
        try:
            model_module = importlib.import_module(f"trajopt.problem_models.{model_name}")
        except ImportError:
            return
        for fn_name in (
            "custom_inputs",
            "custom_subprob_variables",
            "custom_subprob_constraints",
            "custom_subprob_cost",
        ):
            fn = getattr(model_module, fn_name, None)
            if callable(fn):
                fn(self.problem, self)

    # ============================================================
    # PARAMETER UPDATES AND SOLVE
    # ============================================================
    def _set_param(self, param: cp.Parameter, val: np.ndarray) -> None:
        val = np.asarray(val)
        if param is self.zs_m and val.shape != (self.N, self.n):
            val = val.reshape(self.N, self.n)
        param.value = val

    def _update_parameters_from_iterate(self) -> float:
        mission, model, method = self.problem.mission, self.problem.model, self.problem.method
        N, nz = self.N, self.nz
        I = self.problem.I[-1]

        start = time.time()
        Ak, Bk, Bkp, Sk, zs_minus = discretize.compute_linsys_discrete(
            I["zs_ref"], I["us_ref"], I["dts_ref"], self.problem
        )
        prop_time_ms = (time.time() - start) * 1000.0

        dcostdz, dcostdu, cost = convexify.compute_cost(I["ts_ref"], I["zs_ref"], I["us_ref"], self.problem)

        n_ineq_cols = int(self.n_path + self.n_nfz + self.n_aux)
        if n_ineq_cols > 0:
            dgdz, dgdu, g = convexify.compute_path_constraints(I["ts_ref"], I["zs_ref"], I["us_ref"], self.problem)
        else:
            dgdz = dgdu = g = None

        # Dynamics & references
        self._set_param(self.Ak,   Ak)
        self._set_param(self.Bk,   Bk)
        self._set_param(self.Bkp,  Bkp)
        self._set_param(self.Sk,   Sk)
        self._set_param(self.zs_m, zs_minus)

        self.dcostdz.value = dcostdz[:, 0, :]
        self.dcostdu.value = dcostdu[:, 0, :]
        self.cost0.value   = cost[:, 0, 0]
        self.zs_ref.value  = I["zs_ref"]
        self.us_ref.value  = I["us_ref"]
        self.dts_ref.value = I["dts_ref"].reshape(self.N - 1, 1)

        if dgdz is not None:
            self.dgdz.value = dgdz
            self.dgdu.value = dgdu
            self.g0.value   = g

        # Bounds
        if mission.n_init > 0 and self.z1 is not None: self.z1.value = mission.zi
        if mission.n_init_ineq > 0 and self.z1_min is not None and self.z1_max is not None:
            self.z1_min.value = mission.zi_min; self.z1_max.value = mission.zi_max
        if mission.n_term > 0 and self.zN is not None: self.zN.value = mission.zf
        if mission.n_term_ineq > 0 and self.zN_min is not None and self.zN_max is not None:
            self.zN_min.value = mission.zf_min; self.zN_max.value = mission.zf_max
        if mission.n_state > 0 and self.z_min is not None and self.z_max is not None:
            self.z_min.value = mission.z_min; self.z_max.value  = mission.z_max
        if mission.n_ctrl  > 0 and self.u_min is not None and self.u_max is not None:
            self.u_min.value = mission.u_min; self.u_max.value  = mission.u_max
        if mission.n_udot > 0 and self.udot_max is not None:
            self.udot_max.value = mission.udot_max
        if self.free_T:
            self.dts_min.value  = float(method.dts_min)
            self.dts_max.value  = float(method.dts_max)
            self.ddts_max.value = float(method.ddts_max)

        # Weights/duals (ensure shapes, fill any scalars/empties)
        W = I["weights"]
        self.w_cost.value = W.get("w_cost", 1.0)
        self.wtr_z.value  = W.get("wtr_z", 1e-2)
        self.wtr_u.value  = W.get("wtr_u", 1e-2)

        W_path_arr  = tools.ensure_shape(W.get("W_path",  0.0), (self.N,  max(self.n_path, 1)))
        W_nfz_arr   = tools.ensure_shape(W.get("W_nfz",   0.0), (self.N,  max(self.n_nfz,  1)))
        W_aux_arr   = tools.ensure_shape(W.get("W_aux",   0.0), (self.N,  max(self.n_aux,  1)))
        W_term_arr  = tools.ensure_shape(W.get("W_term",  0.0), (max(self.n_term, 1),))
        W_dyn_arr   = tools.ensure_shape(W.get("W_dyn",   0.0), (self.N - 1, max(self.nz, 1)))
        W_plus_arr  = tools.ensure_shape(W.get("W_plus",  0.0), (max(self.Npm, 1), max(self.n_plus,  1)))
        W_minus_arr = tools.ensure_shape(W.get("W_minus", 0.0), (max(self.Npm, 1), max(self.n_minus, 1)))

        self.W_path.value  = W_path_arr
        self.W_nfz.value   = W_nfz_arr
        self.W_aux.value   = W_aux_arr
        self.W_term.value  = W_term_arr
        self.W_dyn.value   = W_dyn_arr
        self.W_plus.value  = W_plus_arr
        self.W_minus.value = W_minus_arr

        # Row weights from elementwise weights (numeric; DCP-safe usage in objective)
        if self.w_path_row is not None: self.w_path_row.value = np.max(W_path_arr, axis=1)
        if self.w_nfz_row  is not None: self.w_nfz_row.value  = np.max(W_nfz_arr,  axis=1)
        if self.w_aux_row  is not None: self.w_aux_row.value  = np.max(W_aux_arr,  axis=1)
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

        return prop_time_ms

    def solve_iteration(self) -> Dict[str, Any]:
        prop_time_ms = self._update_parameters_from_iterate()

        solver_name = self.problem.method.solver_opts.get("solver", "ECOS")
        self.subproblem.solve(solver=solver_name, warm_start=True)

        O = self._collect_outputs(prop_time_ms)
        O = convergence.check_convergence_tolerance(self.problem, self, O)
        O = baseline_autotune(self.problem, {}, O)
        display_baseline_subprob_status(self.problem, {}, O)
        return O

    # ============================================================
    # OUTPUT PACKING
    # ============================================================
    def _collect_outputs(self, prop_time_ms: float) -> Dict[str, Any]:
        mission, model, method = self.problem.mission, self.problem.model, self.problem.method
        N, n, m = self.N, self.n, self.m
        I = self.problem.I[-1]

        dz_val, du_val = self.dz.value, self.du.value
        dt_val = self.dt.value if isinstance(self.dt, cp.expressions.expression.Expression) else self.dt

        O: Dict[str, Any] = {"subprob": self.subproblem}

        if self.subproblem.solver_stats is not None:
            solve_time = getattr(self.subproblem.solver_stats, "solve_time", None)
            setup_time = getattr(self.subproblem.solver_stats, "setup_time", None)
            O["solve_time"] = float(solve_time or 0.0) * 1000.0
            O["parse_time"] = float(setup_time or 0.0) * 1000.0
        else:
            O["solve_time"] = None
            O["parse_time"] = None


        O["dz_s"] = dz_val
        O["du_s"] = du_val
        O["dt_s"] = dt_val

        O["zs_ref"]  = I["zs_ref"]
        O["us_ref"]  = I["us_ref"]
        O["dts_ref"] = I["dts_ref"]
        O["ts_ref"]  = I["ts_ref"]

        O["zs"]  = tools.safe_val(dz_val, rows=N, cols=n) + I["zs_ref"]
        O["us"]  = tools.safe_val(du_val, rows=N, cols=m) + I["us_ref"]
        O["dts"] = tools.safe_val(dt_val).squeeze() + I["dts_ref"].squeeze()
        O["ts"]  = np.concatenate(([0], np.cumsum(O["dts"])))
        O["Ts"]  = float(np.sum(O["dts"]))

        # Discretization model
        O["zs_minus"] = self.zs_m.value
        O["Ak"] = self.Ak.value
        O["Bk"] = self.Bk.value
        O["Bkp"] = self.Bkp.value
        O["Sk"]  = self.Sk.value

        # Path residuals and reference cost
        _, _, cnst_path = convexify.compute_path_constraints(O["ts"], O["zs"], O["us"], self.problem)
        O["cnst_path"] = cnst_path
        O["cost"]      = convexify.compute_cost(O["ts"], O["zs"], O["us"], self.problem)[2].sum().item()

        # Convergence data
        conv = {}
        conv["soln"]    = self.subproblem
        conv["vb_path"] = tools.get_val(self.vb_path,  rows=self.n_path, cols=self.N) if self.vb_path  is not None else np.zeros((self.n_path,  self.N))
        conv["vb_nfz"]  = tools.get_val(self.vb_nfz,   rows=self.n_nfz,  cols=self.N) if self.vb_nfz   is not None else np.zeros((self.n_nfz,   self.N))
        conv["vb_aux"]  = tools.get_val(self.vb_aux,   rows=self.n_aux,  cols=self.N) if self.vb_aux   is not None else np.zeros((self.n_aux,   self.N))
        conv["vb_term"] = tools.get_val(self.vb_term,  rows=self.n_term, cols=1)      if self.vb_term  is not None else np.zeros((self.n_term,  1))
        conv["vb_dyn"]  = tools.get_val(self.vb_dyn_p, rows=self.n_dyn,  cols=self.N-1) - tools.get_val(self.vb_dyn_m, rows=self.n_dyn, cols=self.N-1)
        conv["defect"]  = tools.safe_val(self.dz, rows=N, cols=n) + I["zs_ref"] - self.zs_m.value
        conv["Jtr"]     = ( float(self.wtr_z.value) * np.sum(tools.safe_val(self.dz, rows=N, cols=n)**2)
                          + float(self.wtr_u.value) * np.sum(tools.safe_val(self.du, rows=N, cols=m)**2) )
        conv["cost_ref"] = convexify.compute_cost(I["ts_ref"], I["zs_ref"], I["us_ref"], self.problem)[2].sum().item()

        O["conv_data"]  = conv
        O["weights"]    = I["weights"]
        O["prop_time"]  = prop_time_ms

        return O


# ===========================
# Baseline autotune wrapper
# ===========================
def baseline_autotune(problem, _unused, O: Dict[str, Any]) -> Dict[str, Any]:
    flag = problem.method.bools["flag_autotune"]
    if flag == 1:
        O = hp.autotune1(problem, {}, O)
    elif flag == 2:
        O = hp.autotune2(problem, {}, O)
    elif flag == 3:
        O = hp.autotune3(problem, {}, O)
    return O


# ==========================================
# Iteration status printout (kept)
# ==========================================
def display_baseline_subprob_status(problem, _unused, O: Dict[str, Any]) -> None:
    conv = O.get("conv_data", {})

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
    iter_num    = problem.I[-1].get("iter_num", -1)

    nt    = problem.method.nondim["nt"]
    ncost = problem.method.nondim["ncost"]

    Ts   = O.get("Ts", 0.0)
    cost = O.get("cost", 0.0)

    prop_ms = O.get("prop_time", 0.0)
    solve_ms= float(O.get("solve_time", 0.0) or 0.0)
    parse_ms= float(O.get("parse_time", 0.0) or 0.0)

    print(
        "     {:02d}     |    {:07.1f}   |   {:06.1f}  |   {:06.1f}   |   {:+04.1f}    |      {:+05.1f}      |    {:+05.1f}    |     {:+05.1f}   |    {:s}    |   {:4.2f}   |  {:4.1f}".format(
            int(iter_num),
            float(prop_ms),
            solve_ms,
            parse_ms,
            float(log_dz),
            float(log_vb_ineq),
            float(log_vb_term),
            float(log_vb_dyn),
            str(solve_stat),
            float(Ts * nt),
            float(cost * ncost)
        )
    )