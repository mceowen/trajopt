"""Estimate the integration error on a finer Radau grid."""

import numpy as np
from trajopt.common.interpolation import _polynomial_interpolator
from trajopt.methods.common.discretize import (
    flipped_radau_differential_operator,
    segment_ranges,
)


def estimate_mesh_error(
    mesh, z, nu, dynamics, params, dilation_idx=None, *, extra_points=1
):
    """Patterson--Hager--Rao, doi:10.1002/oca.2114, Sec. 5.1, Eqs. 27--30.

    Integrate dynamics on p+extra_points Radau nodes from each initial state.
    Divide state errors by 1+max(abs(state)) for each segment and component.
    This estimates dynamics error; it does not check path constraints.
    """
    z, nu = np.asarray(z, dtype=float), np.asarray(nu, dtype=float)
    n = len(mesh.tau)
    if mesh.discretize != "ps":
        raise ValueError("Mesh-error estimation requires a pseudospectral mesh")
    if (
        isinstance(extra_points, bool)
        or int(extra_points) != extra_points
        or extra_points < 1
    ):
        raise ValueError("extra_points must be a positive integer")
    if z.ndim != 2 or nu.ndim != 2 or z.shape[0] != n or nu.shape[0] != n:
        raise ValueError("State and control arrays must match the mesh")
    if not np.isfinite(z).all() or not np.isfinite(nu).all():
        raise ValueError("Mesh-error inputs must be finite")
    records = []
    for start, end in segment_ranges(n, mesh.segments):
        p = end - start
        _, enriched, _, D = flipped_radau_differential_operator(p + int(extra_points))
        tau = mesh.tau[start : start + p + 1]
        width = tau[-1] - tau[0]
        check_tau = tau[0] + (enriched + 1) * width / 2
        state = _polynomial_interpolator(tau, z[start : start + p + 1])(check_tau)
        control = (
            np.repeat(nu[start + 1 : start + 2], len(check_tau) - 1, axis=0)
            if p == 1
            else _polynomial_interpolator(tau[1:], nu[start + 1 : start + p + 1])(
                check_tau[1:]
            )
        )
        if dilation_idx is not None:
            control[:, dilation_idx] = nu[start + 1, dilation_idx]
        rhs = np.asarray([dynamics(x, u, params) for x, u in zip(state[1:], control)])
        integrated = z[start] + np.linalg.solve(D[:, 1:], rhs * width / 2)
        error = np.abs(integrated - state[1:])
        relative = error / (1 + np.max(np.abs(state), axis=0))
        if not np.isfinite(relative).all():
            raise ValueError("Nonfinite enriched dynamics or mesh error")
        records.append(
            dict(
                tau=check_tau[1:],
                absolute_error=error,
                relative_error=relative,
                max_relative_error=float(np.max(relative)),
            )
        )
    return dict(
        estimator="Patterson-Hager-Rao enriched flipped Radau",
        extra_points=int(extra_points),
        segments=records,
        max_relative_error=max(r["max_relative_error"] for r in records),
    )
