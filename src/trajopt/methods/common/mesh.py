"""Normalized meshes for transcription and guess sampling."""
from dataclasses import dataclass

import numpy as np

from trajopt.methods.common import discretize


@dataclass(frozen=True)
class Mesh:
    tau: np.ndarray
    discretize: str
    segments: int = 1
    differentiation: np.ndarray | None = None


def make_mesh(num_nodes, options, fcns=None):
    if num_nodes < 2:
        raise ValueError('A trajectory mesh needs at least two nodes')
    kind = getattr(options, 'mode', 'ms')
    if kind == 'ms':
        return Mesh(np.linspace(0, 1, num_nodes), kind)
    if kind != 'ps':
        raise ValueError(f'Unknown discretization {kind!r}')
    segments = getattr(options, 'hp_segments', 1)
    if int(segments) != segments or segments < 1 or (num_nodes-1) % segments:
        raise ValueError('hp_segments must be a positive integer dividing num_nodes - 1')
    segments = int(segments)
    if segments == 1:
        operator = fcns.differential_operator if fcns is not None else discretize.flipped_radau_differential_operator
        _, nodes, _, D = operator(num_nodes-1)
    else:
        operator = fcns.hp_operator if fcns is not None else discretize.flipped_radau_hp_operator
        _, nodes, _, D = operator(num_nodes-1, segments)
    tau = (nodes + 1) / 2
    tau[0], tau[-1] = 0., 1.
    return Mesh(tau, kind, segments, D)
