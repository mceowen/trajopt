"""Radau meshes with different interval widths and polynomial degrees."""

import jax.numpy as jnp
import numpy as np
from trajopt.common.interpolation import _degrees


def segment_ranges(num_nodes, segments):
    ends = np.cumsum([0] + _degrees(segments, num_nodes))
    return tuple(zip(ends[:-1].tolist(), ends[1:].tolist()))


def make_explicit_mesh(num_nodes, definition):
    """Build Radau nodes, weights and derivatives for specified intervals."""
    from trajopt.methods.common.discretize import (
        Mesh,
        flipped_radau_differential_operator,
    )

    hp_mesh = definition
    if set(hp_mesh) != {"knots", "degrees"}:
        raise ValueError("hp_mesh requires knots and degrees")
    knots = np.asarray(hp_mesh["knots"], dtype=float)
    degrees = tuple(hp_mesh["degrees"])
    ranges = segment_ranges(num_nodes, degrees)
    if (
        knots.shape != (len(degrees) + 1,)
        or not np.isfinite(knots).all()
        or knots[0] != 0
        or knots[-1] != 1
        or np.any(np.diff(knots) <= 0)
    ):
        raise ValueError("hp_mesh knots must increase strictly from 0 to 1")
    tau, weights = np.zeros(num_nodes), np.zeros(num_nodes - 1)
    D = np.zeros((num_nodes - 1, num_nodes))
    for h, (a, b) in enumerate(ranges):
        _, local, w, block = flipped_radau_differential_operator(b - a)
        width = knots[h + 1] - knots[h]
        tau[a : b + 1] = knots[h] + (local + 1) * width / 2
        weights[a:b] = w * width / 2
        D[a:b, a : b + 1] = block / width
    tau[0], tau[-1] = 0.0, 1.0
    return Mesh(tau, "ps", D, weights, degrees)


def _control_polynomials(tau, nu, degrees, ranges):
    # Pad shorter segments so JAX can select any segment from one array.
    width = max(degrees)
    indices = np.array(
        [
            [start + min(i + 1, end - start) for i in range(width)]
            for start, end in ranges
        ]
    )
    valid = jnp.asarray(np.arange(width)[None, :] < np.asarray(degrees)[:, None])
    nodes, values = tau[indices], nu[indices]
    differences = nodes[:, :, None] - nodes[:, None, :]
    use = valid[:, :, None] & valid[:, None, :] & ~jnp.eye(width, dtype=bool)[None]
    weights = 1 / jnp.prod(jnp.where(use, differences, 1.0), axis=2)
    return nodes, values, valid, weights


def _polynomial_value(t, nodes, values, valid, weights):
    delta = t - nodes
    exact = jnp.argmin(jnp.where(valid, jnp.abs(delta), jnp.inf))
    w = jnp.where(valid, weights / (delta + 1e-300), 0.0)
    value = jnp.sum(w[:, None] * values, axis=0) / jnp.sum(w)
    # At a node, use its stored value to avoid division by zero.
    return jnp.where(jnp.abs(delta[exact]) < 1e-14, values[exact], value)


def interpolate_controls(tau_nodes, nu_nodes, degrees, dilation_idx, kind):
    """Interpolate controls on a mesh whose segment degrees can differ."""
    ranges = segment_ranges(len(tau_nodes), degrees)
    tau, nu = jnp.asarray(tau_nodes), jnp.asarray(nu_nodes)
    nodes, values, valid, weights = _control_polynomials(tau, nu, degrees, ranges)
    ends = tau[jnp.array([end for start, end in ranges])]
    starts = jnp.array([start + 1 for start, end in ranges])

    def interpolate(t, h=None):
        if h is None:
            h = jnp.clip(jnp.searchsorted(ends, t, side="left"), 0, len(ranges) - 1)
        if kind == "foh":
            k = jnp.clip(jnp.searchsorted(tau, t, side="right") - 1, 0, len(tau) - 2)
            alpha = (t - tau[k]) / (tau[k + 1] - tau[k])
            value = (1 - alpha) * nu[k] + alpha * nu[k + 1]
        else:
            value = _polynomial_value(t, nodes[h], values[h], valid[h], weights[h])
        if dilation_idx is not None:
            value = value.at[dilation_idx].set(nu[starts[h], dilation_idx])
        return value

    return interpolate
