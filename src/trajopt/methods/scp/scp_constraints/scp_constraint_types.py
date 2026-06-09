import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np

from trajopt.methods.scp import pseudospectral
from trajopt.methods.scp.scp_constraints.scp_constraint import SCPConstraint

# ---------------------------------------------------------------------------
# dynamics
# ---------------------------------------------------------------------------

class scp_dynamics(SCPConstraint):
    def compile(self, scp_segment):
        dyn_fcn = jax.jit(self.constraint.fcn_znu)
        second_order = getattr(scp_segment.flags, 'second_order', True)

        df_dz  = jax.jit(jax.jacfwd(self.constraint.fcn_znu, argnums=0))
        df_dnu = jax.jit(jax.jacfwd(self.constraint.fcn_znu, argnums=1))
        self.lin_dyn = lambda z, nu, params: (dyn_fcn(z, nu, params), df_dz(z, nu, params), df_dnu(z, nu, params))

        if scp_segment.flags.discretize == "ps":
            N_col = scp_segment.index_map.N.all - 1
            H = int(getattr(scp_segment.flags, 'hp_segments', 1))

            if H > 1:
                _, etau, _, D_local = pseudospectral.flipped_radau_hp_operator(N_col, H)
                self.ps_D = D_local
                self.ps_hp = H
                self.ps_p = N_col // H
            else:
                _, etau, _, D_np = pseudospectral.flipped_radau_differential_operator(N_col)
                self.ps_D = D_np
                self.ps_hp = 1
                self.ps_p = N_col

            self.ps_etau = etau
            self.ps_tau_norm = (etau + 1.0) / 2.0
            scp_segment.ps_tau_norm = self.ps_tau_norm
            self.dyn_fcn_batched = jax.jit(jax.vmap(dyn_fcn, in_axes=(0, 0, None)))

            if second_order:
                def ps_lagrangian_k(lam_k, z_k, nu_k, params):
                    return -lam_k @ dyn_fcn(z_k, nu_k, params)

                self.ps_cnstr_hessians = jax.jit(jax.vmap(
                    jax.hessian(ps_lagrangian_k, argnums=(1, 2)),
                    in_axes=(0, 0, 0, None),
                ))
            return

        N_grid    = scp_segment.index_map.N.all
        nsub      = 2
        delta_tau = 1.0 / (N_grid - 1)
        dt_rk4    = delta_tau / nsub

        def f_dot(k, tau, z, nu_k, nu_kp, params):
            tau_k  = k / (N_grid - 1)
            tau_kp = (k + 1) / (N_grid - 1)
            a      = (tau_kp - tau) / (tau_kp - tau_k)
            b      = (tau - tau_k)  / (tau_kp - tau_k)
            nu     = a * nu_k + b * nu_kp
            return dyn_fcn(z, nu, params)

        def rk4_step(carry, tau):
            z, k, nu_k, nu_kp, params = carry
            k1 = f_dot(k, tau,            z,                   nu_k, nu_kp, params)
            k2 = f_dot(k, tau + dt_rk4/2, z + (dt_rk4/2) * k1, nu_k, nu_kp, params)
            k3 = f_dot(k, tau + dt_rk4/2, z + (dt_rk4/2) * k2, nu_k, nu_kp, params)
            k4 = f_dot(k, tau + dt_rk4,   z +     dt_rk4 * k3, nu_k, nu_kp, params)
            z_next = z + (dt_rk4 / 6) * (k1 + 2*k2 + 2*k3 + k4)
            return (z_next, k, nu_k, nu_kp, params), None

        def propagate_k(k, z_k, nu_k, nu_kp, params):
            tau_k = k / (N_grid - 1)
            taus  = tau_k + jnp.arange(nsub) * dt_rk4
            carry_init = (z_k, k, nu_k, nu_kp, params)
            (z_kp, _, _, _, _), _ = jax.lax.scan(rk4_step, carry_init, taus)
            return z_kp

        prop_jacobians_k = jax.jacfwd(propagate_k, argnums=(1, 2, 3))

        self.propagate           = jax.jit(jax.vmap(propagate_k,      in_axes=(0, 0, 0, 0, None)))
        self.propagate_jacobians = jax.jit(jax.vmap(prop_jacobians_k, in_axes=(0, 0, 0, 0, None)))

        if second_order:
            def lagrangian_k(k, lam_k, z_k, nu_k, nu_kp, params):
                return -lam_k @ propagate_k(k, z_k, nu_k, nu_kp, params)

            cnstr_hessians_k = jax.hessian(lagrangian_k, argnums=(2, 3, 4))
            self.cnstr_hessians = jax.jit(jax.vmap(cnstr_hessians_k, in_axes=(0, 0, 0, 0, 0, None)))

    def init_penalty(self, scp_segment):
        N   = scp_segment.index_map.N.all
        n_z = scp_segment.index_map.n.z
        self._alloc_penalty(scp_segment, (N - 1, n_z))

    def create_cvxpy_parameters(self, scp_segment):
        N, n_z, n_nu = scp_segment.index_map.N.all, scp_segment.index_map.n.z, scp_segment.index_map.n.nu

        second_order = getattr(scp_segment.flags, 'second_order', True)

        if scp_segment.flags.discretize == "ms":
            scp_segment.cp_params.Ak  = cp.Parameter((N - 1, n_z, n_z),  name="Ak")
            scp_segment.cp_params.Bk  = cp.Parameter((N - 1, n_z, n_nu), name="Bk")
            scp_segment.cp_params.Bkp = cp.Parameter((N - 1, n_z, n_nu), name="Bkp")
            scp_segment.cp_params.z_m = cp.Parameter((N, n_z), name="z_minus")
            if second_order:
                n_w = n_z + n_nu
                scp_segment.cp_params.L = cp.Parameter((N, n_w, n_w), name="L")

        if scp_segment.flags.discretize == "ps":
            N_col = N - 1
            if second_order:
                n_w = n_z + n_nu
                scp_segment.cp_params.L = cp.Parameter((N, n_w, n_w), name="L")

            scp_segment.cp_params.ps_f_ref = cp.Parameter((N_col, n_z),       name="ps_f_ref")
            scp_segment.cp_params.ps_Ac    = cp.Parameter((N_col, n_z, n_z),  name="ps_Ac")
            scp_segment.cp_params.ps_Bc    = cp.Parameter((N_col, n_z, n_nu), name="ps_Bc")

    def create_cvxpy_constraints(self, scp_segment):
        N = scp_segment.index_map.N.all

        if scp_segment.flags.discretize == "ps":
            self._build_ps_dyn_constraints(scp_segment)

        if scp_segment.flags.discretize == "ms":
            vb_dyn = self.vb_var
            scp_segment.cp_dyn_constraints = []

            for k in range(N - 1):
                dz_k   = scp_segment.dz[k]
                dnu_k  = scp_segment.dnu[k]
                dnu_kp = scp_segment.dnu[k + 1]

                Ak  = scp_segment.cp_params.Ak[k]
                Bk  = scp_segment.cp_params.Bk[k]
                Bkp = scp_segment.cp_params.Bkp[k]

                rhs = Ak @ dz_k + Bk @ dnu_k + Bkp @ dnu_kp

                z_ref_prop_kp = scp_segment.cp_params.z_m[k + 1]
                lhs = scp_segment.dz[k + 1] + scp_segment.cp_params.z_ref[k + 1]
                rhs_full = z_ref_prop_kp + rhs + (vb_dyn[k] if vb_dyn is not None else 0)

                cnst = (lhs == rhs_full)
                scp_segment.cp_dyn_constraints.append(cnst)
                scp_segment.cp_constraints.append(cnst)

    def update_cvxpy_parameters(self, scp_segment):
        z_opt  = scp_segment.current_iter_data.z_opt
        nu_opt = scp_segment.current_iter_data.nu_opt

        if scp_segment.flags.discretize == "ms":
            Ak, Bk, Bkp, z_minus = self._compute_linsys_discrete(z_opt, nu_opt, scp_segment)

            scp_segment.cp_params.Ak.value  = Ak
            scp_segment.cp_params.Bk.value  = Bk
            scp_segment.cp_params.Bkp.value = Bkp
            scp_segment.cp_params.z_m.value = z_minus

        if scp_segment.flags.discretize == "ps":
            f_ref_col, Ac_col, Bc_col = self._compute_ps_dynamics_and_jacobians(z_opt, nu_opt, scp_segment)

            scp_segment.cp_params.ps_f_ref.value = f_ref_col
            scp_segment.cp_params.ps_Ac.value    = Ac_col
            scp_segment.cp_params.ps_Bc.value    = Bc_col

            if getattr(scp_segment.flags, 'second_order', True):
                lam_refs = jnp.asarray(scp_segment.lagrangian_duals.dynamics)
                z_col    = jnp.asarray(z_opt[1:])
                nu_col   = jnp.asarray(nu_opt[1:])
                self.ps_H_z, self.ps_H_nu = self.ps_cnstr_hessians(lam_refs, z_col, nu_col, scp_segment.params)

    def accumulate_hessian(self, scp_segment, H):
        n_z = scp_segment.index_map.n.z

        if scp_segment.flags.discretize == "ps":
            H[1:, :n_z, :n_z] += np.asarray(self.ps_H_z[0])
            H[1:, :n_z, n_z:] += np.asarray(self.ps_H_z[1])
            H[1:, n_z:, :n_z] += np.asarray(self.ps_H_nu[0])
            H[1:, n_z:, n_z:] += np.asarray(self.ps_H_nu[1])
            return

        H[:-1, :n_z, :n_z] += np.asarray(self.H_z_k[0])
        H[:-1, :n_z, n_z:] += np.asarray(self.H_z_k[1])
        H[:-1, n_z:, :n_z] += np.asarray(self.H_nu_k[0])
        H[:-1, n_z:, n_z:] += np.asarray(self.H_nu_k[1])
        H[1:,  n_z:, n_z:] += np.asarray(self.H_nu_kp[2])

    def update_current_iter_data(self, scp_segment):
        z_opt  = scp_segment.current_iter_data.z_opt
        nu_opt = scp_segment.current_iter_data.nu_opt

        if scp_segment.flags.discretize == "ps":
            z_jnp  = jnp.asarray(z_opt)
            nu_jnp = jnp.asarray(nu_opt)
            D_jnp  = jnp.asarray(self.ps_D)
            H = self.ps_hp
            p = self.ps_p

            lhs_parts = []
            for h in range(H):
                col_start = h * p
                z_h = z_jnp[col_start:col_start + p + 1, :]
                lhs_parts.append(2.0 * D_jnp @ z_h)
            lhs = jnp.concatenate(lhs_parts, axis=0)

            f_vals = self.dyn_fcn_batched(z_jnp[1:], nu_jnp[1:], scp_segment.params)
            scp_segment.current_iter_data.defect = np.asarray(lhs - f_vals)

        if scp_segment.flags.discretize == "ms":
            ks         = jnp.arange(scp_segment.index_map.N.all - 1)
            z_ref_ks   = jnp.asarray(z_opt[:-1])
            nu_ref_ks  = jnp.asarray(nu_opt[:-1])
            nu_ref_kps = jnp.asarray(nu_opt[1:])
            z_minus    = np.asarray(self.propagate(ks, z_ref_ks, nu_ref_ks, nu_ref_kps, scp_segment.params))
            scp_segment.current_iter_data.defect = z_opt[1:] - z_minus

    def compile_merit_penalty(self, scp_segment):
        if self.W.size == 0:
            return

        if scp_segment.flags.discretize == "ps":
            D_jnp = jnp.asarray(self.ps_D)
            dyn_batched = self.dyn_fcn_batched
            H = self.ps_hp
            p = self.ps_p

            def violation(z, nu, params):
                f_col = dyn_batched(z[1:], nu[1:], params)
                defects = []
                for h in range(H):
                    col_start = h * p
                    z_h = jax.lax.dynamic_slice(z, (col_start, 0), (p + 1, z.shape[1]))
                    defects.append(2.0 * D_jnp @ z_h)
                return jnp.concatenate(defects, axis=0) - f_col

            self._compile_merit_penalty(violation)
            return

        propagate = self.propagate
        ks        = jnp.arange(scp_segment.index_map.N.all - 1)
        def violation(z, nu, params):
            return z[1:] - propagate(ks, z[:-1], nu[:-1], nu[1:], params)
        self._compile_merit_penalty(violation)

    def _compute_linsys_discrete(self, z_ref_np, nu_ref_np, scp_segment):
        segment    = scp_segment
        z_ref_ks   = jnp.asarray(z_ref_np[:-1])
        nu_ref_ks  = jnp.asarray(nu_ref_np[:-1])
        nu_ref_kps = jnp.asarray(nu_ref_np[1:])
        lam_refs   = jnp.asarray(scp_segment.lagrangian_duals.dynamics)
        params     = segment.params

        ks = jnp.arange(segment.index_map.N.all - 1)

        z_minus              = self.propagate(ks, z_ref_ks, nu_ref_ks, nu_ref_kps, params)
        A_jax, B_jax, Bp_jax = self.propagate_jacobians(ks, z_ref_ks, nu_ref_ks, nu_ref_kps, params)

        if getattr(scp_segment.flags, 'second_order', True):
            self.H_z_k, self.H_nu_k, self.H_nu_kp = self.cnstr_hessians(ks, lam_refs, z_ref_ks, nu_ref_ks, nu_ref_kps, params)

        z_ref_0 = z_ref_ks[[0], :]
        return np.asarray(A_jax), np.asarray(B_jax), np.asarray(Bp_jax), np.asarray(jnp.vstack([z_ref_0, z_minus]))

    def _build_ps_dyn_constraints(self, scp_segment):
        N_col = scp_segment.index_map.N.all - 1
        vb_dyn = self.vb_var
        H = self.ps_hp
        p = self.ps_p
        D = self.ps_D

        scp_segment.cp_dyn_constraints = []

        Z = scp_segment.cp_params.z_ref + scp_segment.dz

        for h in range(H):
            col_start = h * p
            Z_h = Z[col_start:col_start + p + 1, :]
            lhs_h = 2.0 * (D @ Z_h)

            for j in range(p):
                k = h * p + j
                rhs_k = (scp_segment.cp_params.ps_f_ref[k]
                         + scp_segment.cp_params.ps_Ac[k] @ scp_segment.dz[k + 1]
                         + scp_segment.cp_params.ps_Bc[k] @ scp_segment.dnu[k + 1]
                         + (vb_dyn[k] if vb_dyn is not None else 0))
                cnst = (lhs_h[j] == rhs_k)
                scp_segment.cp_dyn_constraints.append(cnst)
                scp_segment.cp_constraints.append(cnst)

    def _compute_ps_dynamics_and_jacobians(self, z_ref, nu_ref, scp_segment):
        segment = scp_segment
        N_col   = segment.index_map.N.all - 1
        n_z     = segment.index_map.n.z
        n_nu    = segment.index_map.n.nu
        params  = segment.params

        f_ref_col = np.zeros((N_col, n_z))
        Ac_col    = np.zeros((N_col, n_z, n_z))
        Bc_col    = np.zeros((N_col, n_z, n_nu))

        for k in range(N_col):
            z_k  = np.asarray(z_ref[k + 1])
            nu_k = np.asarray(nu_ref[k + 1])
            fc_k, Ac_k, Bc_k = self.lin_dyn(z_k, nu_k, params)
            f_ref_col[k, :] = np.asarray(fc_k)
            Ac_col[k, :, :] = np.asarray(Ac_k)
            Bc_col[k, :, :] = np.asarray(Bc_k)

        return f_ref_col, Ac_col, Bc_col

# ---------------------------------------------------------------------------
# initial state
# ---------------------------------------------------------------------------

class scp_initial_state(SCPConstraint):
    def init_penalty(self, scp_segment):
        self._alloc_penalty(scp_segment, (1, self.constraint.dimension))

    def create_cvxpy_constraints(self, scp_segment):
        idx  = self.constraint.idx
        expr = scp_segment.dz[0, idx] + scp_segment.cp_params.z_ref[0, idx]
        if self.vb_var is not None:
            expr = expr - self.vb_var[0, :]
        scp_segment.cp_constraints.append(expr == self.constraint.value)

# ---------------------------------------------------------------------------
# final state
# ---------------------------------------------------------------------------

class scp_final_state(SCPConstraint):
    def init_penalty(self, scp_segment):
        self._alloc_penalty(scp_segment, (1, self.constraint.dimension))

    def create_cvxpy_constraints(self, scp_segment):
        idx  = self.constraint.idx
        expr = scp_segment.dz[-1, idx] + scp_segment.cp_params.z_ref[-1, idx]
        if self.vb_var is not None:
            expr = expr - self.vb_var[0, :]
        scp_segment.cp_constraints.append(expr == self.constraint.value)

    def compile_merit_penalty(self, scp_segment):
        if self.W.size == 0:
            return
        idx = jnp.asarray(self.constraint.idx)
        val = jnp.asarray(self.constraint.value)
        def violation(z, nu, params):
            return (z[-1, idx] - val).reshape(1, -1)
        self._compile_merit_penalty(violation)


# ---------------------------------------------------------------------------
# initial control
# ---------------------------------------------------------------------------

class scp_initial_control(SCPConstraint):
    def init_penalty(self, scp_segment):
        self._alloc_penalty(scp_segment, (1, self.constraint.dimension))

    def create_cvxpy_constraints(self, scp_segment):
        idx  = self.constraint.idx
        expr = scp_segment.dnu[0, idx] + scp_segment.cp_params.nu_ref[0, idx]
        if self.vb_var is not None:
            expr = expr - self.vb_var[0, :]
        scp_segment.cp_constraints.append(expr == self.constraint.value)

# ---------------------------------------------------------------------------
# final control
# ---------------------------------------------------------------------------

class scp_final_control(SCPConstraint):
    def init_penalty(self, scp_segment):
        self._alloc_penalty(scp_segment, (1, self.constraint.dimension))

    def create_cvxpy_constraints(self, scp_segment):
        idx  = self.constraint.idx
        expr = scp_segment.dnu[-1, idx] + scp_segment.cp_params.nu_ref[-1, idx]
        if self.vb_var is not None:
            expr = expr - self.vb_var[0, :]
        scp_segment.cp_constraints.append(expr == self.constraint.value)

    def compile_merit_penalty(self, scp_segment):
        if self.W.size == 0:
            return
        idx = jnp.asarray(self.constraint.idx)
        val = jnp.asarray(self.constraint.value)
        def violation(z, nu, params):
            return (nu[-1, idx] - val).reshape(1, -1)
        self._compile_merit_penalty(violation)


# ---------------------------------------------------------------------------
# nonconvex inequality
# ---------------------------------------------------------------------------

class scp_nonconvex_inequality(SCPConstraint):
    nonnegative_dual = True

    def compile(self, scp_segment):
        fcn = self.constraint.fcn_znu
        g      = jax.jit(fcn)
        dg_dz  = jax.jit(jax.jacfwd(fcn, argnums=0))
        dg_dnu = jax.jit(jax.jacfwd(fcn, argnums=1))
        g_batched      = jax.jit(jax.vmap(g,      in_axes=(0, 0, None)))
        dg_dz_batched  = jax.jit(jax.vmap(dg_dz,  in_axes=(0, 0, None)))
        dg_dnu_batched = jax.jit(jax.vmap(dg_dnu, in_axes=(0, 0, None)))

        if getattr(scp_segment.flags, 'second_order', True):
            def lagrangian_k(lam, z, nu, params, fcn=fcn):
                return lam @ fcn(z, nu, params)
            self.lagrangian_hessians = jax.jit(jax.vmap(jax.hessian(lagrangian_k, argnums=(1, 2)), in_axes=(0, 0, 0, None)))

        self.fcn_batched = g_batched

        def g_aff_batched(z, nu, params, g=g_batched, dz=dg_dz_batched, dnu=dg_dnu_batched):
            return g(z, nu, params), dz(z, nu, params), dnu(z, nu, params)
        self.g_aff_batched = g_aff_batched

    def init_penalty(self, scp_segment):
        self.nodes = np.arange(scp_segment.index_map.N.all)
        dim = self.constraint.dimension
        self._alloc_penalty(scp_segment, (len(self.nodes), dim))
        self.lagrangian_dual = np.zeros((len(self.nodes), dim))

    def create_cvxpy_parameters(self, scp_segment):
        if self.shape is None:
            return
        n_z  = scp_segment.index_map.n.z
        n_nu = scp_segment.index_map.n.nu
        dim  = self.constraint.dimension
        nn   = len(self.nodes)
        self.dgdz_param  = cp.Parameter((nn, dim, n_z),  name=f"dgdz_{self.name}")
        self.dgdnu_param = cp.Parameter((nn, dim, n_nu), name=f"dgdnu_{self.name}")
        self.g0_param    = cp.Parameter((nn, dim),       name=f"g0_{self.name}")

    def create_cvxpy_constraints(self, scp_segment):
        if self.vb_var is None:
            return
        self.cp_ineq_constraints = []
        for i, k in enumerate(self.nodes):
            cnst = (
                self.dgdz_param[i] @ scp_segment.dz[k, :]
                + self.dgdnu_param[i] @ scp_segment.dnu[k, :]
                + self.g0_param[i]
                - self.vb_var[i]
                <= 0
            )
            self.cp_ineq_constraints.append(cnst)
            scp_segment.cp_constraints.append(cnst)
            scp_segment.cp_constraints.append(self.vb_var[i] >= 0)

    def update_cvxpy_parameters(self, scp_segment):
        if not hasattr(self, 'g0_param'):
            return
        z      = jnp.asarray(scp_segment.current_iter_data.z_opt)
        nu     = jnp.asarray(scp_segment.current_iter_data.nu_opt)
        params = scp_segment.params
        g, dgdz, dgdnu = self.g_aff_batched(z[self.nodes], nu[self.nodes], params)
        g    = np.asarray(g)
        dgdz = np.asarray(dgdz)
        dgdnu = np.asarray(dgdnu)

        self.g0_param.value    = g
        self.dgdz_param.value  = dgdz
        self.dgdnu_param.value = dgdnu

    def update_current_iter_data(self, scp_segment):
        if not hasattr(self, 'cp_ineq_constraints'):
            return
        z      = jnp.asarray(scp_segment.current_iter_data.z_opt)
        nu     = jnp.asarray(scp_segment.current_iter_data.nu_opt)
        params = scp_segment.params
        self.g_nl = np.asarray(self.fcn_batched(z[self.nodes], nu[self.nodes], params))

        alpha = scp_segment.current_iter_data.get("alpha", 1.0)
        lam   = np.array([c.dual_value for c in self.cp_ineq_constraints])
        self.lagrangian_dual = (1.0 - alpha) * self.lagrangian_dual + alpha * lam

    def accumulate_hessian(self, scp_segment, H):
        if not hasattr(self, 'cp_ineq_constraints'):
            return
        n_z    = scp_segment.index_map.n.z
        z      = jnp.asarray(scp_segment.current_iter_data.z_opt)
        nu     = jnp.asarray(scp_segment.current_iter_data.nu_opt)
        params = scp_segment.params
        lam    = jnp.asarray(self.lagrangian_dual)

        H_z, H_nu = self.lagrangian_hessians(lam, z[self.nodes], nu[self.nodes], params)
        for i, k in enumerate(self.nodes):
            H[k, :n_z, :n_z] += np.asarray(H_z[0][i])
            H[k, :n_z, n_z:] += np.asarray(H_z[1][i])
            H[k, n_z:, :n_z] += np.asarray(H_nu[0][i])
            H[k, n_z:, n_z:] += np.asarray(H_nu[1][i])

    def compile_merit_penalty(self, scp_segment):
        if self.W.size == 0:
            return
        fcn_b = self.fcn_batched
        nodes = jnp.asarray(self.nodes)
        dim   = self.constraint.dimension
        def violation(z, nu, params):
            return jnp.maximum(0.0, fcn_b(z[nodes], nu[nodes], params)).reshape(-1, dim)
        self._compile_merit_penalty(violation)


class scp_initial_nonconvex_inequality(scp_nonconvex_inequality):
    def init_penalty(self, scp_segment):
        self.nodes = np.array([0])
        dim = self.constraint.dimension
        self._alloc_penalty(scp_segment, (len(self.nodes), dim))
        self.lagrangian_dual = np.zeros((len(self.nodes), dim))


class scp_final_nonconvex_inequality(scp_nonconvex_inequality):
    def init_penalty(self, scp_segment):
        self.nodes = np.array([scp_segment.index_map.N.all - 1])
        dim = self.constraint.dimension
        self._alloc_penalty(scp_segment, (len(self.nodes), dim))
        self.lagrangian_dual = np.zeros((len(self.nodes), dim))


class scp_ctcs_nonconvex_inequality(SCPConstraint):

    def create_cvxpy_constraints(self, scp_segment):
        idx_beta = scp_segment.index_map.indices.z.ctcs
        if len(idx_beta) == 0:
            return
        beta_0 = scp_segment.cp_params.z_ref[0, idx_beta] + scp_segment.dz[0, idx_beta]
        scp_segment.cp_constraints.append(beta_0 == 0)

        beta_f = scp_segment.cp_params.z_ref[-1, idx_beta] + scp_segment.dz[-1, idx_beta]
        scp_segment.cp_constraints.append(beta_f <= 0)


# ---------------------------------------------------------------------------
# continuity (cross-segment)
# ---------------------------------------------------------------------------

class scp_continuity(SCPConstraint):

    def init_penalty(self, scp_segment):
        self._scp_segment = scp_segment

    def build_cross_segment(self, scp_segments):
        c = self.constraint
        other = scp_segments[c.segment_name]
        seg = self._scp_segment

        residual = c.residual(other, seg)
        self._alloc_penalty(seg, (1, residual.shape[0]))
        self.create_penalty_parameters(seg)
        self.create_penalty_variables(seg)

        if self.vb_var is not None:
            seg.cp_constraints.append(residual - self.vb_var[0, :] == 0)
            self.add_penalty_cost(seg)
        else:
            seg.cp_constraints.append(residual == 0)


# ---------------------------------------------------------------------------
# convex inequality
# ---------------------------------------------------------------------------

class scp_convex_inequality(SCPConstraint):
    def create_cvxpy_constraints(self, scp_segment):
        params = scp_segment.params
        z_all  = scp_segment.cp_params.z_ref + scp_segment.dz
        nu_all = scp_segment.cp_params.nu_ref + scp_segment.dnu
        N      = scp_segment.index_map.N.all
        expr   = self.constraint.fcn_znu(z_all[:N], nu_all[:N], params)
        scp_segment.cp_constraints.append(expr <= 0)


class scp_initial_convex_inequality(SCPConstraint):
    def create_cvxpy_constraints(self, scp_segment):
        params = scp_segment.params
        z_all  = scp_segment.cp_params.z_ref + scp_segment.dz
        nu_all = scp_segment.cp_params.nu_ref + scp_segment.dnu
        expr   = self.constraint.fcn_znu(z_all[0:1], nu_all[0:1], params)
        scp_segment.cp_constraints.append(expr <= 0)


class scp_final_convex_inequality(SCPConstraint):
    def create_cvxpy_constraints(self, scp_segment):
        params = scp_segment.params
        z_all  = scp_segment.cp_params.z_ref + scp_segment.dz
        nu_all = scp_segment.cp_params.nu_ref + scp_segment.dnu
        expr   = self.constraint.fcn_znu(z_all[-1:], nu_all[-1:], params)
        scp_segment.cp_constraints.append(expr <= 0)


# ---------------------------------------------------------------------------
# state limits
# ---------------------------------------------------------------------------

class scp_state_limits(SCPConstraint):
    def create_cvxpy_constraints(self, scp_segment):
        idx_state = scp_segment.index_map.indices.z.state
        for k in range(scp_segment.index_map.N.all):
            x_k = scp_segment.cp_params.z_ref[k, idx_state] + scp_segment.dz[k, idx_state]
            if self.constraint.lower_idx:
                scp_segment.cp_constraints.append(x_k[self.constraint.lower_idx] >= self.constraint.lower_value)
            if self.constraint.upper_idx:
                scp_segment.cp_constraints.append(x_k[self.constraint.upper_idx] <= self.constraint.upper_value)


class scp_initial_state_limits(SCPConstraint):
    def create_cvxpy_constraints(self, scp_segment):
        idx_state = scp_segment.index_map.indices.z.state
        x_k = scp_segment.cp_params.z_ref[0, idx_state] + scp_segment.dz[0, idx_state]
        if self.constraint.lower_idx:
            scp_segment.cp_constraints.append(x_k[self.constraint.lower_idx] >= self.constraint.lower_value)
        if self.constraint.upper_idx:
            scp_segment.cp_constraints.append(x_k[self.constraint.upper_idx] <= self.constraint.upper_value)


class scp_final_state_limits(SCPConstraint):
    def create_cvxpy_constraints(self, scp_segment):
        idx_state = scp_segment.index_map.indices.z.state
        x_k = scp_segment.cp_params.z_ref[-1, idx_state] + scp_segment.dz[-1, idx_state]
        if self.constraint.lower_idx:
            scp_segment.cp_constraints.append(x_k[self.constraint.lower_idx] >= self.constraint.lower_value)
        if self.constraint.upper_idx:
            scp_segment.cp_constraints.append(x_k[self.constraint.upper_idx] <= self.constraint.upper_value)


# ---------------------------------------------------------------------------
# control limits
# ---------------------------------------------------------------------------

class scp_control_limits(SCPConstraint):
    def create_cvxpy_constraints(self, scp_segment):
        idx_ctrl = scp_segment.index_map.indices.nu.control
        for k in range(scp_segment.index_map.N.all):
            u_k = scp_segment.cp_params.nu_ref[k, idx_ctrl] + scp_segment.dnu[k, idx_ctrl]
            if self.constraint.lower_idx:
                scp_segment.cp_constraints.append(u_k[self.constraint.lower_idx] >= self.constraint.lower_value)
            if self.constraint.upper_idx:
                scp_segment.cp_constraints.append(u_k[self.constraint.upper_idx] <= self.constraint.upper_value)


class scp_initial_control_limits(SCPConstraint):
    def create_cvxpy_constraints(self, scp_segment):
        idx_ctrl = scp_segment.index_map.indices.nu.control
        u_k = scp_segment.cp_params.nu_ref[0, idx_ctrl] + scp_segment.dnu[0, idx_ctrl]
        if self.constraint.lower_idx:
            scp_segment.cp_constraints.append(u_k[self.constraint.lower_idx] >= self.constraint.lower_value)
        if self.constraint.upper_idx:
            scp_segment.cp_constraints.append(u_k[self.constraint.upper_idx] <= self.constraint.upper_value)


class scp_final_control_limits(SCPConstraint):
    def create_cvxpy_constraints(self, scp_segment):
        idx_ctrl = scp_segment.index_map.indices.nu.control
        u_k = scp_segment.cp_params.nu_ref[-1, idx_ctrl] + scp_segment.dnu[-1, idx_ctrl]
        if self.constraint.lower_idx:
            scp_segment.cp_constraints.append(u_k[self.constraint.lower_idx] >= self.constraint.lower_value)
        if self.constraint.upper_idx:
            scp_segment.cp_constraints.append(u_k[self.constraint.upper_idx] <= self.constraint.upper_value)


# ---------------------------------------------------------------------------
# control rate limit
# ---------------------------------------------------------------------------

class scp_control_rate_limit(SCPConstraint):
    def create_cvxpy_constraints(self, scp_segment):
        idx_ctrl = scp_segment.index_map.indices.nu.control
        value    = self.constraint.value
        M_sel    = self.constraint.M_select
        for k in range(scp_segment.index_map.N.all - 1):
            du_k = (
                scp_segment.cp_params.nu_ref[k + 1, idx_ctrl] + scp_segment.dnu[k + 1, idx_ctrl]
                - (scp_segment.cp_params.nu_ref[k, idx_ctrl] + scp_segment.dnu[k, idx_ctrl])
            )
            dt_k = (scp_segment.t_ref[k + 1, 0] + scp_segment.dt[k + 1, 0]) - (scp_segment.t_ref[k, 0] + scp_segment.dt[k, 0])
            scp_segment.cp_constraints.append(M_sel @ du_k <= dt_k * np.concatenate([value, value]))

# ---------------------------------------------------------------------------
# final time
# ---------------------------------------------------------------------------

class scp_final_time(SCPConstraint):
    def create_cvxpy_parameters(self, scp_segment):
        scp_segment.cp_params.T_min  = cp.Parameter(nonneg=True, name="T_min")
        scp_segment.cp_params.T_max  = cp.Parameter(nonneg=True, name="T_max")
        scp_segment.cp_params.dt_min = cp.Parameter(nonneg=True, name="dt_min")
        scp_segment.cp_params.dt_max = cp.Parameter(nonneg=True, name="dt_max")

    def create_cvxpy_constraints(self, scp_segment):
        N = scp_segment.index_map.N.all
        scp_segment.cp_constraints.append(scp_segment.dt[0, 0] == 0)

        if scp_segment.flags.discretize == "ps":
            tau = scp_segment.cp_params.tau
            for k in range(1, N - 1):
                scp_segment.cp_constraints.append(scp_segment.dt[k, 0] == tau[k] * scp_segment.dt[N - 1, 0])
        else:
            for k in range(N - 1):
                t_k          = scp_segment.t_ref[k, 0] + scp_segment.dt[k, 0]
                t_kp         = scp_segment.t_ref[k + 1, 0] + scp_segment.dt[k + 1, 0]
                t_interval_k = t_kp - t_k
                scp_segment.cp_constraints.append(t_interval_k <= scp_segment.cp_params.dt_max)
                scp_segment.cp_constraints.append(t_interval_k >= scp_segment.cp_params.dt_min)

        scp_segment.cp_constraints.append(scp_segment.cp_params.T_min <= scp_segment.t_ref[-1, 0] + scp_segment.dt[-1, 0])
        scp_segment.cp_constraints.append(scp_segment.t_ref[-1, 0] + scp_segment.dt[-1, 0] <= scp_segment.cp_params.T_max)

    def update_cvxpy_parameters(self, scp_segment):
        if self.constraint.lower is not None:
            scp_segment.cp_params.T_min.value  = float(self.constraint.lower)
            scp_segment.cp_params.dt_min.value = float(self.constraint.dt_min)
        if self.constraint.upper is not None:
            scp_segment.cp_params.T_max.value  = float(self.constraint.upper)
            scp_segment.cp_params.dt_max.value = float(self.constraint.dt_max)
