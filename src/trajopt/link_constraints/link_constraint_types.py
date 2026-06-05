from typing import Any

import cvxpy as cp
import jax.numpy as jnp
import numpy as np


class continuity:
    def __init__(self, cnstr_config: dict, segment1, segment2) -> None:
        self.name      = cnstr_config.name
        self.type      = "continuity"
        self.segment1  = segment1
        self.segment2  = segment2
        self.segment1_name = cnstr_config.segment1
        self.segment2_name = cnstr_config.segment2
        self.penalty   = cnstr_config.get("penalty")

    def _selectors(self, index_map: Any) -> tuple[np.ndarray, np.ndarray]:
        idx    = index_map.indices
        z_sel  = np.concatenate([idx.z.state, idx.z.time])
        nu_sel = np.asarray(idx.nu.control)
        return z_sel, nu_sel

    def residual(self, subprob1: Any, subprob2: Any):
        z_sel, nu_sel = self._selectors(subprob1.index_map)
        z1  = subprob1.cp_params.z_ref[-1, z_sel]   + subprob1.dz[-1, z_sel]
        z2  = subprob2.cp_params.z_ref[0, z_sel]    + subprob2.dz[0, z_sel]
        nu1 = subprob1.cp_params.nu_ref[-1, nu_sel] + subprob1.dnu[-1, nu_sel]
        nu2 = subprob2.cp_params.nu_ref[0, nu_sel]  + subprob2.dnu[0, nu_sel]
        return cp.hstack([z1 - z2, nu1 - nu2])

    def residual_jax(self, index_map: Any):
        z_sel, nu_sel = self._selectors(index_map)

        def fcn(z1, nu1, z2, nu2):
            return jnp.concatenate([z1[-1, z_sel] - z2[0, z_sel],
                                    nu1[-1, nu_sel] - nu2[0, nu_sel]])

        return fcn
