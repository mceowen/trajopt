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


def make_mesh(num_nodes, options, fcns=None):
    if num_nodes < 2:
        raise ValueError('A trajectory mesh needs at least two nodes')
    kind = getattr(options, 'mode', 'ms')
    if kind == 'ms':
        return Mesh(np.linspace(0, 1, num_nodes), kind)
    if kind != 'ps':
        raise ValueError(f'Unknown discretization {kind!r}')
    operator = fcns.differential_operator if fcns is not None else discretize.flipped_radau_differential_operator
    _, nodes, w, D = operator(num_nodes-1)
    tau = (nodes + 1) / 2
    tau[0], tau[-1] = 0., 1.
    return Mesh(tau, kind, D, w / 2)
