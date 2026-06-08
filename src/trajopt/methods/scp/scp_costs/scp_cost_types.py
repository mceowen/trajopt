import cvxpy as cp
import jax.numpy as jnp
import numpy as np
from trajopt.methods.scp.scp_costs.scp_cost import SCPCost

class _scp_jax_cost(SCPCost):
    def create_cvxpy_cost(self, scp_segment):
        if _first_nonconvex_cost(scp_segment) is not self.cost:
            return
        scp_segment.cp_cost += (
            cp.sum(scp_segment.cp_params.cost0)
            + cp.sum(cp.multiply(scp_segment.cp_params.dcostdx, scp_segment.dz))
            + cp.sum(cp.multiply(scp_segment.cp_params.dcostdu, scp_segment.dnu))
        )

    def update_cvxpy_parameters(self, scp_segment):
        if _first_nonconvex_cost(scp_segment) is not self.cost:
            return
        z_opt  = scp_segment.current_iter_data.z_opt
        nu_opt = scp_segment.current_iter_data.nu_opt
        cost, dcostdx, dcostdu = compute_nonconvex_costs(z_opt, nu_opt, scp_segment.segment, scp_segment)
        scp_segment.cp_params.dcostdx.value = dcostdx.squeeze(axis=1)
        scp_segment.cp_params.dcostdu.value = dcostdu.squeeze(axis=1)
        scp_segment.cp_params.cost0.value   = cost.squeeze(axis=1)

    def accumulate_hessian(self, scp_segment, H):
        if _first_nonconvex_cost(scp_segment) is not self.cost:
            return
        z_opt  = scp_segment.current_iter_data.z_opt
        nu_opt = scp_segment.current_iter_data.nu_opt
        n_z    = scp_segment.index_map.n.z

        H_cost_z, H_cost_nu, H_cost_znu = compute_nonconvex_cost_hessians(z_opt, nu_opt, scp_segment.segment, scp_segment)
        H[:, :n_z, :n_z] += H_cost_z
        H[:, n_z:, n_z:] += H_cost_nu
        H[:, :n_z, n_z:] += H_cost_znu
        H[:, n_z:, :n_z] += np.transpose(H_cost_znu, (0, 2, 1))

class scp_running(_scp_jax_cost):
    def merit_cost(self, scp_segment):
        fcn_b = self.cost.fcn_batched

        def eval_fn(z, nu, params):
            return jnp.sum(fcn_b(z, nu, params))
        return eval_fn


class scp_terminal(_scp_jax_cost):
    def merit_cost(self, scp_segment):
        fcn_b = self.cost.fcn_batched
        nodes = jnp.asarray(self.cost.nodes)

        def eval_fn(z, nu, params):
            return jnp.sum(fcn_b(z[nodes], nu[nodes], params))
        return eval_fn


class scp_convex_terminal(SCPCost):
    def create_cvxpy_cost(self, scp_segment):
        idx_state = scp_segment.index_map.indices.z.state
        idx_ctrl  = scp_segment.index_map.indices.nu.control
        for k in np.atleast_1d(self.cost.nodes):
            x_k = scp_segment.cp_params.z_ref[k, idx_state] + scp_segment.dz[k, idx_state]
            u_k = scp_segment.cp_params.nu_ref[k, idx_ctrl] + scp_segment.dnu[k, idx_ctrl]
            scp_segment.cp_cost += self.cost.w * self.cost.fcn_dim(x_k, u_k, 0, scp_segment.params)


class scp_convex_running(SCPCost):
    def create_cvxpy_cost(self, scp_segment):
        idx_state = scp_segment.index_map.indices.z.state
        idx_ctrl  = scp_segment.index_map.indices.nu.control
        for k in range(scp_segment.index_map.N.all):
            if k in self.cost.nodes:
                x_k = scp_segment.cp_params.z_ref[k, idx_state] + scp_segment.dz[k, idx_state]
                u_k = scp_segment.cp_params.nu_ref[k, idx_ctrl] + scp_segment.dnu[k, idx_ctrl]
                scp_segment.cp_cost += self.cost.w * self.cost.fcn_dim(x_k, u_k, 0, scp_segment.params)

class scp_min_time(SCPCost):
    def create_cvxpy_cost(self, scp_segment):
        if bool(scp_segment.flags.free_final_time):
            s = scp_segment.t_ref[:, 0] + scp_segment.dt[:, 0]
            scp_segment.cp_cost += cp.sum(s)

    def merit_cost(self, scp_segment):
        dil_idx = jnp.array(scp_segment.index_map.indices.nu.dilation_factor)

        def eval_fn(z, nu, params):
            return jnp.sum(nu[:, dil_idx[0]])
        return eval_fn

class scp_min_norm_terminal(SCPCost):
    def create_cvxpy_cost(self, scp_segment):
        zf     = scp_segment.cp_params.z_ref[-1] + scp_segment.dz[-1]
        target = self.cost.value if self.cost.value is not None else np.zeros(len(self.cost.idx))
        scp_segment.cp_cost += cp.norm(zf[self.cost.idx] - target)

class scp_terminal_state(SCPCost):
    def create_cvxpy_cost(self, scp_segment):
        zf = scp_segment.cp_params.z_ref[-1] + scp_segment.dz[-1]
        scp_segment.cp_cost += self.cost.sign * zf[self.cost.idx]

    def merit_cost(self, scp_segment):
        sign = self.cost.sign
        idx  = jnp.array(self.cost.idx)

        def eval_fn(z, nu, params):
            return sign * jnp.sum(z[-1, idx])
        return eval_fn

class scp_regularization(SCPCost):
    def create_cvxpy_cost(self, scp_segment):
        if self.cost.set == "control":
            traj = scp_segment.cp_params.nu_ref + scp_segment.dnu
        elif self.cost.set == "state":
            traj = scp_segment.cp_params.z_ref + scp_segment.dz
        else:
            return

        if self.cost.norm_type == "l2":
            scp_segment.cp_cost += self.cost.w * cp.sum_squares(traj)
        elif self.cost.norm_type == "l1":
            scp_segment.cp_cost += self.cost.w * cp.norm1(traj)

    def merit_cost(self, scp_segment):
        w         = self.cost.w
        norm_type = self.cost.norm_type
        is_nu     = self.cost.set == "control"

        def eval_fn(z, nu, params):
            traj = nu if is_nu else z
            if norm_type == "l2":
                return w * jnp.sum(traj ** 2)
            return w * jnp.sum(jnp.abs(traj))
        return eval_fn

class scp_rate_regularization(SCPCost):
    def create_cvxpy_cost(self, scp_segment):
        if self.cost.set == "control":
            traj = scp_segment.cp_params.nu_ref + scp_segment.dnu
        elif self.cost.set == "state":
            traj = scp_segment.cp_params.z_ref + scp_segment.dz
        else:
            return

        delta = traj[1:, self.cost.idx] - traj[:-1, self.cost.idx]

        if self.cost.norm_type == "l2":
            scp_segment.cp_cost += self.cost.w * cp.sum_squares(delta)
        elif self.cost.norm_type == "l1":
            scp_segment.cp_cost += self.cost.w * cp.norm1(delta)

    def merit_cost(self, scp_segment):
        w         = self.cost.w
        norm_type = self.cost.norm_type
        is_nu     = self.cost.set == "control"

        def eval_fn(z, nu, params):
            traj  = nu if is_nu else z
            delta = traj[1:] - traj[:-1]
            if norm_type == "l2":
                return w * jnp.sum(delta ** 2)
            return w * jnp.sum(jnp.abs(delta))
        return eval_fn

def _first_nonconvex_cost(scp_segment):
    for scp_cost in scp_segment.costs.values():
        if isinstance(scp_cost, _scp_jax_cost):
            return scp_cost.cost
    return None


def compute_nonconvex_costs(z, nu, segment, scp_segment):
    N         = segment.index_map.N.all
    n_z       = segment.index_map.n.z
    n_nu       = segment.index_map.n.nu

    cost     = np.zeros((N, 1))
    dcostdz  = np.zeros((N, 1, n_z))
    dcostdnu = np.zeros((N, 1, n_nu))

    params = segment.params
    nonconvex_costs = [c for c in segment.costs.values() if c.type == "nonconvex"]
    terminal_costs  = [c for c in segment.costs.values() if c.type == "terminal"]
    running_costs   = [c for c in segment.costs.values() if c.type == "running"]

    if len(nonconvex_costs) + len(terminal_costs) + len(running_costs) == 0:
        return cost, dcostdz, dcostdnu

    z_jax  = jnp.asarray(z)
    nu_jax = jnp.asarray(nu)

    for cost_fn in nonconvex_costs + running_costs:
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

def compute_nonconvex_cost_hessians(z, nu, segment, scp_segment):
    N = segment.index_map.N.all
    n_z = segment.index_map.n.z
    n_nu = segment.index_map.n.nu

    H_cost_z = np.zeros((N, n_z, n_z))
    H_cost_nu = np.zeros((N, n_nu, n_nu))
    H_cost_znu = np.zeros((N, n_z, n_nu))

    params = segment.params
    nonconvex_costs = [c for c in segment.costs.values() if c.type == "nonconvex"]
    terminal_costs = [c for c in segment.costs.values() if c.type == "terminal"]
    running_costs = [c for c in segment.costs.values() if c.type == "running"]

    if len(nonconvex_costs) + len(terminal_costs) + len(running_costs) == 0:
        return H_cost_z, H_cost_nu, H_cost_znu

    z_jax = jnp.asarray(z)
    nu_jax = jnp.asarray(nu)

    for cost_fn in nonconvex_costs + running_costs:
        w = getattr(cost_fn, 'w', 1.0)
        d2z = cost_fn.d2fcn_dz2_batched(z_jax, nu_jax, params)
        d2nu = cost_fn.d2fcn_dnu2_batched(z_jax, nu_jax, params)
        d2znu = cost_fn.d2fcn_dzdnu_batched(z_jax, nu_jax, params)
        H_cost_z += w * np.asarray(d2z).reshape(N, n_z, n_z)
        H_cost_nu += w * np.asarray(d2nu).reshape(N, n_nu, n_nu)
        H_cost_znu += w * np.asarray(d2znu).reshape(N, n_z, n_nu)

    for cost_fn in terminal_costs:
        nodes = np.atleast_1d(cost_fn.nodes)
        z_nodes = z_jax[nodes]
        nu_nodes = nu_jax[nodes]
        d2z = cost_fn.d2fcn_dz2_batched(z_nodes, nu_nodes, params)
        d2nu = cost_fn.d2fcn_dnu2_batched(z_nodes, nu_nodes, params)
        d2znu = cost_fn.d2fcn_dzdnu_batched(z_nodes, nu_nodes, params)
        for i, k in enumerate(nodes):
            H_cost_z[k] += np.asarray(d2z[i]).reshape(n_z, n_z)
            H_cost_nu[k] += np.asarray(d2nu[i]).reshape(n_nu, n_nu)
            H_cost_znu[k] += np.asarray(d2znu[i]).reshape(n_z, n_nu)

    return H_cost_z, H_cost_nu, H_cost_znu
