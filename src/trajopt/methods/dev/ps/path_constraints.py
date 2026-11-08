"""Path constraints at both sides of pseudospectral segment boundaries."""

import numpy as np
from trajopt.common.interpolation import _polynomial_interpolator

from trajopt.methods.common.discretize import segment_ranges


def path_sample_maps(mesh):
    """Include the opening value of each internal segment."""
    n = len(mesh.tau)
    nodes = np.arange(n)
    if mesh.discretize != "ps" or mesh.segments == 1:
        return nodes, None, None
    ranges = segment_ranges(n, mesh.segments)[1:]
    if not ranges:
        return nodes, None, None
    starts = [a for a, b in ranges]
    control = np.zeros((len(starts), n))
    dilation = np.zeros_like(control)
    for i, (start, end) in enumerate(ranges):
        p = end - start
        local = mesh.tau[start + 1 : start + p + 1]
        control[i, start + 1 : start + p + 1] = (
            np.ones(1)
            if p == 1
            else _polynomial_interpolator(local, np.eye(p))(mesh.tau[start])
        )
        dilation[i, start + 1] = 1.0
    return (
        np.concatenate((nodes, starts)),
        np.vstack((np.eye(n), control)),
        np.vstack((np.eye(n), dilation)),
    )


def path_samples(subproblem, z, nu):
    """Apply the same linear sampling maps to arrays or CVXPY expressions."""
    nodes, control, dilation = subproblem.path_sample_maps
    if control is None:
        return z, nu
    mask = np.zeros(nu.shape[1])
    mask[subproblem.index_map.indices.nu.dilation_factor] = 1.0
    return z[nodes], control @ nu @ np.diag(1 - mask) + dilation @ nu @ np.diag(mask)


def constraint_samples(constraint, subproblem, z, nu):
    if getattr(constraint, "hp_path_samples", False):
        return path_samples(subproblem, z, nu)
    return z[constraint.nodes], nu[constraint.nodes]


def accumulate_path_hessian(subproblem, H, H_z, H_nu):
    """Add the sampled Hessian while keeping only same-node blocks."""
    n_z = subproblem.index_map.n.z
    n_nu = subproblem.current_iter_data.nu_opt.shape[1]
    nodes, control, dilation = subproblem.path_sample_maps
    for i, k in enumerate(nodes):
        weights = np.repeat(control[i, :, None], n_nu, axis=1)
        weights[:, subproblem.index_map.indices.nu.dilation_factor] = dilation[
            i, :, None
        ]
        H[k, :n_z, :n_z] += np.asarray(H_z[0][i])
        H[k, :n_z, n_z:] += np.asarray(H_z[1][i]) * weights[k]
        H[k, n_z:, :n_z] += weights[k, :, None] * np.asarray(H_nu[0][i])
        H[:, n_z:, n_z:] += (
            weights[:, :, None] * np.asarray(H_nu[1][i]) * weights[:, None, :]
        )
