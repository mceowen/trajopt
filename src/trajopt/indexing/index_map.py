from collections.abc import Callable

import numpy as np
from jax import Array

from trajopt.utils.tools import AttrDict


class IndexMap:
    """Owns the augmented-vector layout z = [x, t, beta], nu = [u, s] and packs/unpacks it."""

    def __init__(self, config: AttrDict) -> None:
        """Read the core state/control dimensions and the time grid from the config."""
        problem_config = config.problem

        n_state   = max(max(group["idx"]) for group in problem_config["state"].values()) + 1
        n_control = max(max(group["idx"]) for group in problem_config["control"].values()) + 1

        self.n = AttrDict({"state": n_state, "control": n_control})
        self.N = AttrDict({"time_grid": config.method["time_grid"]})

        self.set_augmented_dims(n_ctcs=0)

    def set_augmented_dims(self, n_ctcs: int) -> None:
        """Build the augmented z = [x, t, beta] and nu = [u, s] layout for a given CTCS size."""
        n_state, n_control = self.n.state, self.n.control
        n_time = 1
        n_z    = n_state + n_time + n_ctcs
        n_nu   = n_control + 1

        self.indices = AttrDict({
            "z": AttrDict({
                "state": np.arange(0, n_state),
                "time":  np.arange(n_state, n_state + n_time),
                "ctcs":  np.arange(n_state + n_time, n_z),
            }),
            "nu": AttrDict({
                "control":         np.arange(0, n_control),
                "dilation_factor": np.arange(n_control, n_nu),
            }),
        })

        self.n.time = n_time
        self.n.ctcs = n_ctcs
        self.n.z    = n_z
        self.n.nu   = n_nu

############################################################
# INDEX PACKING / UNPACKING UTILITIES
############################################################

    def unpack_z(self, z: np.ndarray | Array) -> tuple[np.ndarray | Array, np.ndarray | Array, np.ndarray | Array]:
        """Unpack augmented state z = [x, t, beta]. Supports z shaped (n_z,) or (N, n_z)."""
        z_ndim = getattr(z, "ndim", np.ndim(z))
        if z_ndim == 2:
            x = z[:, self.indices.z.state]
            t = z[:, self.indices.z.time]
            beta = z[:, self.indices.z.ctcs]
        else:
            x = z[self.indices.z.state]
            t = z[self.indices.z.time]
            beta = z[self.indices.z.ctcs]
        return x, t, beta

    def unpack_nu(self, nu: np.ndarray | Array) -> tuple[np.ndarray | Array, np.ndarray | Array]:
        """Unpack augmented control nu = [u, s]. Supports nu shaped (n_nu,) or (N, n_nu)."""
        nu_ndim = getattr(nu, "ndim", np.ndim(nu))
        if nu_ndim == 2:
            u = nu[:, self.indices.nu.control]
            s = nu[:, self.indices.nu.dilation_factor]
        else:
            u = nu[self.indices.nu.control]
            s = nu[self.indices.nu.dilation_factor]
        return u, s

    def unpack_znu(self, z: np.ndarray | Array, nu: np.ndarray | Array) -> tuple[np.ndarray | Array, ...]:
        """Unpack augmented state and control into physically meaningful variables."""
        x, t, beta = self.unpack_z(z)
        u, s       = self.unpack_nu(nu)
        return x, t, beta, u, s

    def pack_z(self, x: np.ndarray | Array, t: np.ndarray | Array, beta: np.ndarray | Array) -> np.ndarray:
        """Pack physical variables into augmented state z = [x, t, beta]."""
        x_ndim = getattr(x, "ndim", np.ndim(x))
        t_ndim = getattr(t, "ndim", np.ndim(t))
        beta_ndim = getattr(beta, "ndim", np.ndim(beta))

        if x_ndim == 2 or t_ndim == 2 or beta_ndim == 2:
            if x_ndim == 2:
                n_time = x.shape[0]
            elif t_ndim == 2:
                n_time = t.shape[0]
            else:
                n_time = beta.shape[0]

            z = np.zeros((n_time, self.n.z))
            z[:, self.indices.z.state] = x
            z[:, self.indices.z.time] = t
            z[:, self.indices.z.ctcs] = beta
        else:
            z = np.zeros(self.n.z)
            z[self.indices.z.state] = x
            z[self.indices.z.time] = t
            z[self.indices.z.ctcs] = beta
        return z

    def pack_nu(self, u: np.ndarray | Array, s: float | np.ndarray | Array) -> np.ndarray:
        """Pack physical variables into augmented control nu = [u, s]."""
        u_ndim = getattr(u, "ndim", np.ndim(u))
        s_ndim = getattr(s, "ndim", np.ndim(s))

        if u_ndim == 2 or s_ndim == 2:
            n_time = u.shape[0] if u_ndim == 2 else s.shape[0]
            nu = np.zeros((n_time, self.n.nu))
            nu[:, self.indices.nu.control] = u
            nu[:, self.indices.nu.dilation_factor] = s
        else:
            nu = np.zeros(self.n.nu)
            nu[self.indices.nu.control] = u
            nu[self.indices.nu.dilation_factor] = s
        return nu

    def pack_znu(
        self,
        x: np.ndarray | Array,
        t: np.ndarray | Array,
        beta: np.ndarray | Array,
        u: np.ndarray | Array,
        s: float | np.ndarray | Array,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pack physical variables into augmented state and control."""
        z  = self.pack_z(x, t, beta)
        nu = self.pack_nu(u, s)
        return z, nu

    def wrap_txu_fcn(self, f_phys: Callable) -> Callable:
        """Wrap f_phys(t, x, u, params) to be callable as f_alg(z, nu, params)."""
        def f_alg(z, nu, params):
            x, t, beta, u, s = self.unpack_znu(z, nu)
            return f_phys(t, x, u, params)

        return f_alg

    def evaluate_f_phys(self, f_phys: Callable, z: np.ndarray | Array, nu: np.ndarray | Array, params: AttrDict) -> Array:
        """Evaluate f_phys(t, x, u, params) from augmented variables (z, nu)."""
        x, t, beta, u, s = self.unpack_znu(z, nu)
        return f_phys(t, x, u, params)