import cvxpy as cp
import jax
import numpy as np

from trajopt.methods.common.scp.constraint import SCPConstraint

#jax.config.update("jax_compilation_cache_dir", "/absolute/path/to/jax_cache")
#jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
#jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)

# ---------------------------------------------------------------------------
# dynamics
# ---------------------------------------------------------------------------

class scp_dynamics(SCPConstraint):
    def compile(self, scp_subproblem):
        self.dyn_fcn = jax.jit(self.constraint.fcn_znu)
        scp_subproblem.fcns.discretize.dynamics_compile(self, scp_subproblem)

    def init_penalty(self, scp_subproblem):
        N   = scp_subproblem.index_map.N.all
        n_z = scp_subproblem.index_map.n.z
        self._alloc_penalty(scp_subproblem, (N - 1, n_z))
        self.lagrangian_dual = np.zeros((N - 1, n_z))

    def create_cvxpy_parameters(self, scp_subproblem):
        scp_subproblem.fcns.discretize.dynamics_create_cvxpy_parameters(self, scp_subproblem)

    def create_cvxpy_constraints(self, scp_subproblem):
        scp_subproblem.fcns.discretize.dynamics_create_cvxpy_constraints(self, scp_subproblem)

    def update_cvxpy_parameters(self, scp_subproblem):
        scp_subproblem.fcns.discretize.dynamics_update_cvxpy_parameters(self, scp_subproblem)

    def update_current_iter_data(self, scp_subproblem):
        scp_subproblem.fcns.discretize.dynamics_update_current_iter_data(self, scp_subproblem)

        if hasattr(scp_subproblem, 'cp_dyn_constraints') and scp_subproblem.cp_dyn_constraints:
            alpha = scp_subproblem.current_iter_data.get("alpha", 1.0)
            lam = np.array([c.dual_value for c in scp_subproblem.cp_dyn_constraints])
            self.lagrangian_dual = (1.0 - alpha) * self.lagrangian_dual + alpha * lam

# ---------------------------------------------------------------------------
# initial state
# ---------------------------------------------------------------------------

class scp_initial_state(SCPConstraint):
    def init_penalty(self, scp_subproblem):
        self._alloc_penalty(scp_subproblem, (1, self.constraint.dimension))

    def create_cvxpy_constraints(self, scp_subproblem):
        idx  = self.constraint.idx
        expr = scp_subproblem.dz[0, idx] + scp_subproblem.cp_params.z_ref[0, idx]
        if self.penalties.vb_var is not None:
            expr = expr - self.penalties.vb_var[0, :]
        scp_subproblem.cp_constraints.append(expr == self.constraint.value)

# ---------------------------------------------------------------------------
# final state
# ---------------------------------------------------------------------------

class scp_final_state(SCPConstraint):
    def init_penalty(self, scp_subproblem):
        self._alloc_penalty(scp_subproblem, (1, self.constraint.dimension))

    def create_cvxpy_constraints(self, scp_subproblem):
        idx  = self.constraint.idx
        expr = scp_subproblem.dz[-1, idx] + scp_subproblem.cp_params.z_ref[-1, idx]
        if self.penalties.vb_var is not None:
            expr = expr - self.penalties.vb_var[0, :]
        scp_subproblem.cp_constraints.append(expr == self.constraint.value)


# ---------------------------------------------------------------------------
# initial control
# ---------------------------------------------------------------------------

class scp_initial_control(SCPConstraint):
    def init_penalty(self, scp_subproblem):
        self._alloc_penalty(scp_subproblem, (1, self.constraint.dimension))

    def create_cvxpy_constraints(self, scp_subproblem):
        idx  = self.constraint.idx
        expr = scp_subproblem.dnu[0, idx] + scp_subproblem.cp_params.nu_ref[0, idx]
        if self.penalties.vb_var is not None:
            expr = expr - self.penalties.vb_var[0, :]
        scp_subproblem.cp_constraints.append(expr == self.constraint.value)

# ---------------------------------------------------------------------------
# final control
# ---------------------------------------------------------------------------

class scp_final_control(SCPConstraint):
    def init_penalty(self, scp_subproblem):
        self._alloc_penalty(scp_subproblem, (1, self.constraint.dimension))

    def create_cvxpy_constraints(self, scp_subproblem):
        idx  = self.constraint.idx
        expr = scp_subproblem.dnu[-1, idx] + scp_subproblem.cp_params.nu_ref[-1, idx]
        if self.penalties.vb_var is not None:
            expr = expr - self.penalties.vb_var[0, :]
        scp_subproblem.cp_constraints.append(expr == self.constraint.value)


# ---------------------------------------------------------------------------
# nonconvex inequality
# ---------------------------------------------------------------------------

class scp_nonconvex_inequality(SCPConstraint):
    nonnegative_dual = True
    cp_ineq_constraints = None

    def compile(self, scp_subproblem):
        scp_subproblem.fcns.convexify.compile_affine(self, scp_subproblem)

    def init_penalty(self, scp_subproblem):
        self.nodes = np.arange(scp_subproblem.index_map.N.all)
        dim = self.constraint.dimension
        self._alloc_penalty(scp_subproblem, (len(self.nodes), dim))
        self.lagrangian_dual = np.zeros((len(self.nodes), dim))

    def create_cvxpy_parameters(self, scp_subproblem):
        scp_subproblem.fcns.convexify.create_cvxpy_parameters_affine(self, scp_subproblem)

    def create_cvxpy_constraints(self, scp_subproblem):
        scp_subproblem.fcns.convexify.create_cvxpy_constraints_affine_inequality(self, scp_subproblem)

    def update_cvxpy_parameters(self, scp_subproblem):
        scp_subproblem.fcns.convexify.update_cvxpy_parameters_affine(self, scp_subproblem)

    def update_current_iter_data(self, scp_subproblem):
        scp_subproblem.fcns.convexify.update_current_iter_data_affine_inequality(self, scp_subproblem)


class scp_initial_nonconvex_inequality(scp_nonconvex_inequality):
    def init_penalty(self, scp_subproblem):
        self.nodes = np.array([0])
        dim = self.constraint.dimension
        self._alloc_penalty(scp_subproblem, (len(self.nodes), dim))
        self.lagrangian_dual = np.zeros((len(self.nodes), dim))


class scp_final_nonconvex_inequality(scp_nonconvex_inequality):
    def init_penalty(self, scp_subproblem):
        self.nodes = np.array([scp_subproblem.index_map.N.all - 1])
        dim = self.constraint.dimension
        self._alloc_penalty(scp_subproblem, (len(self.nodes), dim))
        self.lagrangian_dual = np.zeros((len(self.nodes), dim))


class scp_ctcs_nonconvex_inequality(SCPConstraint):

    def create_cvxpy_constraints(self, scp_subproblem):
        idx_beta = scp_subproblem.index_map.indices.z.ctcs
        if len(idx_beta) == 0:
            return
        beta_0 = scp_subproblem.cp_params.z_ref[0, idx_beta] + scp_subproblem.dz[0, idx_beta]
        scp_subproblem.cp_constraints.append(beta_0 == 0)

        beta_f = scp_subproblem.cp_params.z_ref[-1, idx_beta] + scp_subproblem.dz[-1, idx_beta]
        scp_subproblem.cp_constraints.append(beta_f <= 0)

# ---------------------------------------------------------------------------
# nonconvex equality
# ---------------------------------------------------------------------------

class scp_nonconvex_equality(SCPConstraint):
    cp_eq_constraints = None

    def compile(self, scp_subproblem):
        scp_subproblem.fcns.convexify.compile_affine(self, scp_subproblem)

    def init_penalty(self, scp_subproblem):
        self.nodes = np.arange(scp_subproblem.index_map.N.all)
        dim = self.constraint.dimension
        self._alloc_penalty(scp_subproblem, (len(self.nodes), dim))
        self.lagrangian_dual = np.zeros((len(self.nodes), dim))

    def create_cvxpy_parameters(self, scp_subproblem):
        scp_subproblem.fcns.convexify.create_cvxpy_parameters_affine(self, scp_subproblem)

    def create_cvxpy_constraints(self, scp_subproblem):
        scp_subproblem.fcns.convexify.create_cvxpy_constraints_affine_equality(self, scp_subproblem)

    def update_cvxpy_parameters(self, scp_subproblem):
        scp_subproblem.fcns.convexify.update_cvxpy_parameters_affine(self, scp_subproblem)

    def update_current_iter_data(self, scp_subproblem):
        scp_subproblem.fcns.convexify.update_current_iter_data_affine_equality(self, scp_subproblem)


class scp_initial_nonconvex_equality(scp_nonconvex_equality):
    def init_penalty(self, scp_subproblem):
        self.nodes = np.array([0])
        dim = self.constraint.dimension
        self._alloc_penalty(scp_subproblem, (len(self.nodes), dim))
        self.lagrangian_dual = np.zeros((len(self.nodes), dim))


class scp_final_nonconvex_equality(scp_nonconvex_equality):
    def init_penalty(self, scp_subproblem):
        self.nodes = np.array([scp_subproblem.index_map.N.all - 1])
        dim = self.constraint.dimension
        self._alloc_penalty(scp_subproblem, (len(self.nodes), dim))
        self.lagrangian_dual = np.zeros((len(self.nodes), dim))


# ---------------------------------------------------------------------------
# continuity (cross-phase)
# ---------------------------------------------------------------------------

class scp_full_continuity(SCPConstraint):

    def init_penalty(self, scp_subproblem):
        self._scp_subproblem = scp_subproblem

    def build_cross_phase(self, scp_subproblems):
        c = self.constraint
        other = scp_subproblems[c.phase_name]
        sub = self._scp_subproblem

        residual = c.residual(other, sub)
        self._alloc_penalty(sub, (1, residual.shape[0]))
        self.create_penalty_parameters(sub)
        self.create_penalty_variables(sub)

        if self.penalties.vb_var is not None:
            sub.cp_constraints.append(residual - self.penalties.vb_var[0, :] == 0)
            self.add_penalty_cost(sub)
        else:
            sub.cp_constraints.append(residual == 0)


class scp_state_continuity(scp_full_continuity):
    pass


class scp_control_continuity(scp_full_continuity):
    pass


class scp_time_continuity(scp_full_continuity):
    pass


# ---------------------------------------------------------------------------
# convex inequality
# ---------------------------------------------------------------------------

class scp_convex_inequality(SCPConstraint):
    def create_cvxpy_constraints(self, scp_subproblem):
        params = scp_subproblem.params
        z_all  = scp_subproblem.cp_params.z_ref + scp_subproblem.dz
        nu_all = scp_subproblem.cp_params.nu_ref + scp_subproblem.dnu
        N      = scp_subproblem.index_map.N.all
        expr   = self.constraint.fcn_znu(z_all[:N], nu_all[:N], params)
        scp_subproblem.cp_constraints.append(expr <= 0)


class scp_initial_convex_inequality(SCPConstraint):
    def create_cvxpy_constraints(self, scp_subproblem):
        params = scp_subproblem.params
        z_all  = scp_subproblem.cp_params.z_ref + scp_subproblem.dz
        nu_all = scp_subproblem.cp_params.nu_ref + scp_subproblem.dnu
        expr   = self.constraint.fcn_znu(z_all[0:1], nu_all[0:1], params)
        scp_subproblem.cp_constraints.append(expr <= 0)


class scp_final_convex_inequality(SCPConstraint):
    def create_cvxpy_constraints(self, scp_subproblem):
        params = scp_subproblem.params
        z_all  = scp_subproblem.cp_params.z_ref + scp_subproblem.dz
        nu_all = scp_subproblem.cp_params.nu_ref + scp_subproblem.dnu
        expr   = self.constraint.fcn_znu(z_all[-1:], nu_all[-1:], params)
        scp_subproblem.cp_constraints.append(expr <= 0)


# ---------------------------------------------------------------------------
# state limits
# ---------------------------------------------------------------------------

class scp_state_limits(SCPConstraint):
    def create_cvxpy_constraints(self, scp_subproblem):
        idx_state = scp_subproblem.index_map.indices.z.state
        for k in range(scp_subproblem.index_map.N.all):
            x_k = scp_subproblem.cp_params.z_ref[k, idx_state] + scp_subproblem.dz[k, idx_state]
            if self.constraint.lower_idx:
                scp_subproblem.cp_constraints.append(x_k[self.constraint.lower_idx] >= self.constraint.lower_value)
            if self.constraint.upper_idx:
                scp_subproblem.cp_constraints.append(x_k[self.constraint.upper_idx] <= self.constraint.upper_value)


class scp_initial_state_limits(SCPConstraint):
    def create_cvxpy_constraints(self, scp_subproblem):
        idx_state = scp_subproblem.index_map.indices.z.state
        x_k = scp_subproblem.cp_params.z_ref[0, idx_state] + scp_subproblem.dz[0, idx_state]
        if self.constraint.lower_idx:
            scp_subproblem.cp_constraints.append(x_k[self.constraint.lower_idx] >= self.constraint.lower_value)
        if self.constraint.upper_idx:
            scp_subproblem.cp_constraints.append(x_k[self.constraint.upper_idx] <= self.constraint.upper_value)


class scp_final_state_limits(SCPConstraint):
    def create_cvxpy_constraints(self, scp_subproblem):
        idx_state = scp_subproblem.index_map.indices.z.state
        x_k = scp_subproblem.cp_params.z_ref[-1, idx_state] + scp_subproblem.dz[-1, idx_state]
        if self.constraint.lower_idx:
            scp_subproblem.cp_constraints.append(x_k[self.constraint.lower_idx] >= self.constraint.lower_value)
        if self.constraint.upper_idx:
            scp_subproblem.cp_constraints.append(x_k[self.constraint.upper_idx] <= self.constraint.upper_value)


# ---------------------------------------------------------------------------
# control limits
# ---------------------------------------------------------------------------

class scp_control_limits(SCPConstraint):
    def create_cvxpy_constraints(self, scp_subproblem):
        idx_ctrl = scp_subproblem.index_map.indices.nu.control
        for k in range(scp_subproblem.index_map.N.all):
            u_k = scp_subproblem.cp_params.nu_ref[k, idx_ctrl] + scp_subproblem.dnu[k, idx_ctrl]
            if self.constraint.lower_idx:
                scp_subproblem.cp_constraints.append(u_k[self.constraint.lower_idx] >= self.constraint.lower_value)
            if self.constraint.upper_idx:
                scp_subproblem.cp_constraints.append(u_k[self.constraint.upper_idx] <= self.constraint.upper_value)


class scp_initial_control_limits(SCPConstraint):
    def create_cvxpy_constraints(self, scp_subproblem):
        idx_ctrl = scp_subproblem.index_map.indices.nu.control
        u_k = scp_subproblem.cp_params.nu_ref[0, idx_ctrl] + scp_subproblem.dnu[0, idx_ctrl]
        if self.constraint.lower_idx:
            scp_subproblem.cp_constraints.append(u_k[self.constraint.lower_idx] >= self.constraint.lower_value)
        if self.constraint.upper_idx:
            scp_subproblem.cp_constraints.append(u_k[self.constraint.upper_idx] <= self.constraint.upper_value)


class scp_final_control_limits(SCPConstraint):
    def create_cvxpy_constraints(self, scp_subproblem):
        idx_ctrl = scp_subproblem.index_map.indices.nu.control
        u_k = scp_subproblem.cp_params.nu_ref[-1, idx_ctrl] + scp_subproblem.dnu[-1, idx_ctrl]
        if self.constraint.lower_idx:
            scp_subproblem.cp_constraints.append(u_k[self.constraint.lower_idx] >= self.constraint.lower_value)
        if self.constraint.upper_idx:
            scp_subproblem.cp_constraints.append(u_k[self.constraint.upper_idx] <= self.constraint.upper_value)


# ---------------------------------------------------------------------------
# control rate limit
# ---------------------------------------------------------------------------

class scp_control_rate_limit(SCPConstraint):
    def create_cvxpy_constraints(self, scp_subproblem):
        if scp_subproblem.index_map.N.all < 2:
            return
        idx_ctrl = scp_subproblem.index_map.indices.nu.control
        value    = self.constraint.value
        M_sel    = self.constraint.M_select
        u = scp_subproblem.cp_params.nu_ref[:, idx_ctrl] + scp_subproblem.dnu[:, idx_ctrl]
        t = scp_subproblem.t_ref[:, 0] + scp_subproblem.dt[:, 0]
        limits = np.concatenate([value, value])
        rhs = cp.reshape(t[1:] - t[:-1], (-1, 1), order="C") @ limits[None, :]
        scp_subproblem.cp_constraints.append((u[1:] - u[:-1]) @ M_sel.T <= rhs)


class scp_control_accel_limit(SCPConstraint):
    """|u_{k+1} - 2 u_k + u_{k-1}| <= value * dt_{k-1} * dt_k, bilinear rhs linearized for DPP."""

    def create_cvxpy_parameters(self, scp_subproblem):
        N = scp_subproblem.index_map.N.all
        self.dt_ref_param  = cp.Parameter((N - 1,), name=f"dt_ref_{self.name}",      value=np.zeros(N - 1))
        self.dt_prod_param = cp.Parameter((N - 2,), name=f"dt_ref_prod_{self.name}", value=np.zeros(N - 2))

    def create_cvxpy_constraints(self, scp_subproblem):
        idx_ctrl = scp_subproblem.index_map.indices.nu.control
        value    = self.constraint.value
        M_sel    = self.constraint.M_select
        N        = scp_subproblem.index_map.N.all

        if N < 3:
            return
        u = scp_subproblem.cp_params.nu_ref[:, idx_ctrl] + scp_subproblem.dnu[:, idx_ctrl]
        d2u = u[2:] - 2 * u[1:-1] + u[:-2]
        # Linearize adjacent interval products using the same reference grid.
        d_interval = scp_subproblem.dt[1:, 0] - scp_subproblem.dt[:-1, 0]
        dt_prod = (
            self.dt_prod_param
            + cp.multiply(self.dt_ref_param[:-1], d_interval[1:])
            + cp.multiply(self.dt_ref_param[1:], d_interval[:-1])
        )
        limits = np.concatenate([value, value])
        rhs = cp.reshape(dt_prod, (-1, 1), order="C") @ limits[None, :]
        scp_subproblem.cp_constraints.append(d2u @ M_sel.T <= rhs)

    def update_cvxpy_parameters(self, scp_subproblem):
        t_ref  = np.asarray(scp_subproblem.t_ref)[:, 0]
        dt_ref = np.maximum(np.diff(t_ref), 0.0)

        self.dt_ref_param.value  = dt_ref
        self.dt_prod_param.value = dt_ref[:-1] * dt_ref[1:]

# ---------------------------------------------------------------------------
# final time
# ---------------------------------------------------------------------------

class scp_initial_time(SCPConstraint):
    """Does not add a constraint. The start time is applied when the grid is built."""


class scp_final_time(SCPConstraint):
    def create_cvxpy_parameters(self, scp_subproblem):
        scp_subproblem.cp_params.T_min  = cp.Parameter(nonneg=True, name="T_min")
        scp_subproblem.cp_params.T_max  = cp.Parameter(nonneg=True, name="T_max")
        scp_subproblem.cp_params.dt_min = cp.Parameter(nonneg=True, name="dt_min")
        scp_subproblem.cp_params.dt_max = cp.Parameter(nonneg=True, name="dt_max")

    def create_cvxpy_constraints(self, scp_subproblem):
        # a fixed grid holds constant times, which already meet these bounds
        if not scp_subproblem.free_final_time:
            return

        scp_subproblem.fcns.discretize.final_time_interval_constraints(self, scp_subproblem)

        scp_subproblem.cp_constraints.append(scp_subproblem.cp_params.T_min <= scp_subproblem.t_ref[-1, 0] + scp_subproblem.dt[-1, 0])
        scp_subproblem.cp_constraints.append(scp_subproblem.t_ref[-1, 0] + scp_subproblem.dt[-1, 0] <= scp_subproblem.cp_params.T_max)

    def update_cvxpy_parameters(self, scp_subproblem):
        if self.constraint.lower is not None:
            scp_subproblem.cp_params.T_min.value  = float(self.constraint.lower)
            scp_subproblem.cp_params.dt_min.value = float(self.constraint.dt_min)
        if self.constraint.upper is not None:
            scp_subproblem.cp_params.T_max.value  = float(self.constraint.upper)
            scp_subproblem.cp_params.dt_max.value = float(self.constraint.dt_max)
