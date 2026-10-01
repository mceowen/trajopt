import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
from trajopt.methods.common.scp.cost import SCPCost


class scp_nonconvex_running(SCPCost):

    def create_cvxpy_cost(self, scp_subproblem):
        idx_rc = scp_subproblem.index_map.indices.z.running_cost
        if len(idx_rc) == 0 or self.cost.gamma_idx is None:
            return
        gamma_z_idx = idx_rc[self.cost.gamma_idx]

        gamma_0 = scp_subproblem.cp_params.z_ref[0, gamma_z_idx] + scp_subproblem.dz[0, gamma_z_idx]
        scp_subproblem.cp_constraints.append(gamma_0 == 0)

        gamma_f = scp_subproblem.cp_params.z_ref[-1, gamma_z_idx] + scp_subproblem.dz[-1, gamma_z_idx]
        scp_subproblem.cp_cost += self.cost.w * gamma_f


class scp_nonconvex_terminal(SCPCost):
    def create_cvxpy_cost(self, scp_subproblem):
        if _first_nonconvex_terminal_cost(scp_subproblem) is not self.cost:
            return
        scp_subproblem.cp_cost += (
            cp.sum(scp_subproblem.cp_params.cost0)
            + cp.sum(cp.multiply(scp_subproblem.cp_params.dcostdx, scp_subproblem.dz))
            + cp.sum(cp.multiply(scp_subproblem.cp_params.dcostdu, scp_subproblem.dnu))
        )

    def update_cvxpy_parameters(self, scp_subproblem):
        if _first_nonconvex_terminal_cost(scp_subproblem) is not self.cost:
            return
        z_opt  = scp_subproblem.current_iter_data.z_opt
        nu_opt = scp_subproblem.current_iter_data.nu_opt
        cost, dcostdx, dcostdu = compute_nonconvex_terminal_costs(z_opt, nu_opt, scp_subproblem.phase, scp_subproblem)
        scp_subproblem.cp_params.dcostdx.value = dcostdx.squeeze(axis=1)
        scp_subproblem.cp_params.dcostdu.value = dcostdu.squeeze(axis=1)
        scp_subproblem.cp_params.cost0.value   = cost.squeeze(axis=1)


class scp_nonconvex_minimax(SCPCost):
    def __init__(self, cost, scp_subproblem):
        super().__init__(cost, scp_subproblem)

        fcn = self.cost.fcn_znu
        g      = jax.jit(fcn)
        dg_dz  = jax.jit(jax.jacfwd(fcn, argnums=0))
        dg_dnu = jax.jit(jax.jacfwd(fcn, argnums=1))
        self.g_batched      = jax.jit(jax.vmap(g,      in_axes=(0, 0, None)))
        self.dg_dz_batched  = jax.jit(jax.vmap(dg_dz,  in_axes=(0, 0, None)))
        self.dg_dnu_batched = jax.jit(jax.vmap(dg_dnu, in_axes=(0, 0, None)))

        self.nodes = np.atleast_1d(cost.nodes)

    def create_cvxpy_cost(self, scp_subproblem):
        n_z  = scp_subproblem.index_map.n.z
        n_nu = scp_subproblem.index_map.n.nu
        nn   = len(self.nodes)

        out = self.cost.fcn_znu(
            jnp.ones(n_z), jnp.ones(n_nu), scp_subproblem.params
        )
        dim = jnp.atleast_1d(out).shape[0]

        self.ub_var         = cp.Variable( dim,             name=f"ub_{self.name}")
        self.g0_param       = cp.Parameter((nn, dim),       name=f"mm_g0_{self.name}")
        self.dgdz_param     = cp.Parameter((nn, dim, n_z),  name=f"mm_dgdz_{self.name}")
        self.dgdnu_param    = cp.Parameter((nn, dim, n_nu), name=f"mm_dgdnu_{self.name}")

        self.cp_ineq_constraints = []
        for i, k in enumerate(self.nodes):
            g_lin = (
                self.g0_param[i]
                + self.dgdz_param[i] @ scp_subproblem.dz[k, :]
                + self.dgdnu_param[i] @ scp_subproblem.dnu[k, :]
            )
            cnst = (g_lin <= self.ub_var)
            self.cp_ineq_constraints.append(cnst)
            scp_subproblem.cp_constraints.append(cnst)

        scp_subproblem.cp_cost += self.cost.w * cp.sum(self.ub_var)

    def update_cvxpy_parameters(self, scp_subproblem):
        z      = jnp.asarray(scp_subproblem.current_iter_data.z_opt)
        nu     = jnp.asarray(scp_subproblem.current_iter_data.nu_opt)
        params = scp_subproblem.params
        nn     = len(self.nodes)
        n_z    = scp_subproblem.index_map.n.z
        n_nu   = scp_subproblem.index_map.n.nu
        dim    = self.g0_param.shape[1]

        g     = np.asarray(self.g_batched(z[self.nodes], nu[self.nodes], params)).reshape(nn, dim)
        dgdz  = np.asarray(self.dg_dz_batched(z[self.nodes], nu[self.nodes], params)).reshape(nn, dim, n_z)
        dgdnu = np.asarray(self.dg_dnu_batched(z[self.nodes], nu[self.nodes], params)).reshape(nn, dim, n_nu)

        self.g0_param.value    = g
        self.dgdz_param.value  = dgdz
        self.dgdnu_param.value = dgdnu


class scp_convex_terminal(SCPCost):
    def create_cvxpy_cost(self, scp_subproblem):
        idx_state = scp_subproblem.index_map.indices.z.state
        idx_ctrl  = scp_subproblem.index_map.indices.nu.control
        for k in np.atleast_1d(self.cost.nodes):
            x_k = scp_subproblem.cp_params.z_ref[k, idx_state] + scp_subproblem.dz[k, idx_state]
            u_k = scp_subproblem.cp_params.nu_ref[k, idx_ctrl] + scp_subproblem.dnu[k, idx_ctrl]
            scp_subproblem.cp_cost += self.cost.w * self.cost.fcn_dim(x_k, u_k, 0, scp_subproblem.params)


class scp_convex_running(SCPCost):
    def create_cvxpy_cost(self, scp_subproblem):
        idx_state = scp_subproblem.index_map.indices.z.state
        idx_ctrl  = scp_subproblem.index_map.indices.nu.control
        for k in range(scp_subproblem.index_map.N.all):
            if k in self.cost.nodes:
                x_k = scp_subproblem.cp_params.z_ref[k, idx_state] + scp_subproblem.dz[k, idx_state]
                u_k = scp_subproblem.cp_params.nu_ref[k, idx_ctrl] + scp_subproblem.dnu[k, idx_ctrl]
                scp_subproblem.cp_cost += self.cost.w * self.cost.fcn_dim(x_k, u_k, 0, scp_subproblem.params)

class scp_min_time(SCPCost):
    def create_cvxpy_cost(self, scp_subproblem):
        if scp_subproblem.free_final_time:
            scp_subproblem.cp_cost += cp.sum(scp_subproblem.t_ref[-1] + scp_subproblem.dt[-1])

class scp_min_norm_terminal(SCPCost):
    def create_cvxpy_cost(self, scp_subproblem):
        zf     = scp_subproblem.cp_params.z_ref[-1] + scp_subproblem.dz[-1]
        target = self.cost.value if self.cost.value is not None else np.zeros(len(self.cost.idx))
        scp_subproblem.cp_cost += cp.norm(zf[self.cost.idx] - target)

class scp_final_state(SCPCost):
    def create_cvxpy_cost(self, scp_subproblem):
        zf = scp_subproblem.cp_params.z_ref[-1] + scp_subproblem.dz[-1]
        scp_subproblem.cp_cost += self.cost.w * cp.sum(zf[self.cost.idx])

class scp_final_control(SCPCost):
    def create_cvxpy_cost(self, scp_subproblem):
        nuf = scp_subproblem.cp_params.nu_ref[-1] + scp_subproblem.dnu[-1]
        scp_subproblem.cp_cost += self.cost.w * cp.sum(nuf[self.cost.idx])

class scp_regularization(SCPCost):
    def create_cvxpy_cost(self, scp_subproblem):
        if self.cost.set == "control":
            traj = scp_subproblem.cp_params.nu_ref + scp_subproblem.dnu
        elif self.cost.set == "state":
            traj = scp_subproblem.cp_params.z_ref + scp_subproblem.dz
        else:
            return

        if self.cost.norm_type == "l2":
            scp_subproblem.cp_cost += self.cost.w * cp.sum_squares(traj)
        elif self.cost.norm_type == "l1":
            scp_subproblem.cp_cost += self.cost.w * cp.norm1(traj)

class scp_rate_regularization(SCPCost):
    def create_cvxpy_cost(self, scp_subproblem):
        if self.cost.set == "control":
            traj = scp_subproblem.cp_params.nu_ref + scp_subproblem.dnu
        elif self.cost.set == "state":
            traj = scp_subproblem.cp_params.z_ref + scp_subproblem.dz
        else:
            return

        delta = traj[1:, self.cost.idx] - traj[:-1, self.cost.idx]

        if self.cost.norm_type == "l2":
            scp_subproblem.cp_cost += self.cost.w * cp.sum_squares(delta)
        elif self.cost.norm_type == "l1":
            scp_subproblem.cp_cost += self.cost.w * cp.norm1(delta)

def _first_nonconvex_terminal_cost(scp_subproblem):
    for scp_cost in scp_subproblem.costs.values():
        if isinstance(scp_cost, scp_nonconvex_terminal):
            return scp_cost.cost
    return None


def compute_nonconvex_terminal_costs(z, nu, phase, scp_subproblem):
    N    = phase.index_map.N.all
    n_z  = phase.index_map.n.z
    n_nu = phase.index_map.n.nu

    cost     = np.zeros((N, 1))
    dcostdz  = np.zeros((N, 1, n_z))
    dcostdnu = np.zeros((N, 1, n_nu))

    params = phase.params
    nonconvex_costs = [c for c in phase.costs.values() if c.type == "nonconvex"]
    terminal_costs  = [c for c in phase.costs.values() if c.type == "nonconvex_terminal"]

    if len(nonconvex_costs) + len(terminal_costs) == 0:
        return cost, dcostdz, dcostdnu

    z_jax  = jnp.asarray(z)
    nu_jax = jnp.asarray(nu)

    for cost_fn in nonconvex_costs:
        w = getattr(cost_fn, 'w', 1.0)
        f_batch, dfdx_batch, dfdu_batch = cost_fn.g_aff_batched(z_jax, nu_jax, params)

        f_np    = np.asarray(f_batch).reshape(N, -1)
        dfdx_np = np.asarray(dfdx_batch).reshape(N, -1, n_z)
        dfdu_np = np.asarray(dfdu_batch).reshape(N, -1, n_nu)

        cost[:, 0] += w * np.sum(f_np, axis=1)
        dcostdz[:, 0, :] += w * np.sum(dfdx_np, axis=1)
        dcostdnu[:, 0, :] += w * np.sum(dfdu_np, axis=1)

    for cost_fn in terminal_costs:
        nodes = np.atleast_1d(cost_fn.nodes)
        z_nodes  = z_jax[nodes]
        nu_nodes = nu_jax[nodes]
        f_batch, dfdx_batch, dfdu_batch = cost_fn.g_aff_batched(z_nodes, nu_nodes, params)

        f_np    = np.asarray(f_batch).reshape(len(nodes), -1)
        dfdx_np = np.asarray(dfdx_batch).reshape(len(nodes), -1, n_z)
        dfdu_np = np.asarray(dfdu_batch).reshape(len(nodes), -1, n_nu)

        for i, k in enumerate(nodes):
            cost[k, 0]       += np.sum(f_np[i])
            dcostdz[k, 0, :]  += np.sum(dfdx_np[i], axis=0)
            dcostdnu[k, 0, :] += np.sum(dfdu_np[i], axis=0)

    return cost, dcostdz, dcostdnu
