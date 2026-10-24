"""Normalized meshes for transcription and guess sampling."""
from dataclasses import dataclass

import numpy as np

from trajopt.methods.common import discretize


@dataclass(frozen=True)
class Mesh:
    tau: np.ndarray
    discretize: str
    differentiation: np.ndarray | None = None
    weights: np.ndarray | None = None  # quadrature weights on [0, 1] at the collocation nodes tau[1:]
    segments: int = 1  # hp segments of equal order; knots at node indices h * (N-1) / segments


def make_mesh(num_nodes, options, fcns=None, segments=1):
    if num_nodes < 2:
        raise ValueError('A trajectory mesh needs at least two nodes')
    kind = getattr(options, 'mode', 'ms')
    if kind == 'ms':
        return Mesh(np.linspace(0, 1, num_nodes), kind)
    if kind != 'ps':
        raise ValueError(f'Unknown discretization {kind!r}')
    if int(segments) != segments or segments < 1 or (num_nodes-1) % segments:
        raise ValueError(f"hp_segments={segments} must be a positive integer "
                         f"dividing num_nodes - 1 = {num_nodes - 1}")
    segments = int(segments)
    operator = fcns.differential_operator if fcns is not None else discretize.flipped_radau_hp_operator
    _, nodes, w, D = operator(num_nodes-1, segments)
    tau = (nodes + 1) / 2
    tau[0], tau[-1] = 0., 1.
    return Mesh(tau, kind, D, w / 2, segments)
