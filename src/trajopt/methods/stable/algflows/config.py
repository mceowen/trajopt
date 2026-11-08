"""Read and check algorithm-flow refinement settings."""

from collections.abc import Mapping
import numpy as np


def refinement_options(options):
    defaults = dict(
        tolerance=1e-6, min_degree=3, max_degree=10, max_meshes=8, max_nodes=1500
    )
    if not isinstance(options, Mapping) or set(options) - defaults.keys():
        raise ValueError(
            "refinement accepts tolerance, min_degree, max_degree, max_meshes, max_nodes"
        )
    defaults.update(options)
    for key in ("min_degree", "max_degree", "max_meshes", "max_nodes"):
        value = defaults[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"refinement.{key} must be a positive integer")
    if defaults["min_degree"] < 2 or defaults["max_degree"] < defaults["min_degree"]:
        raise ValueError("refinement requires 2 <= min_degree <= max_degree")
    if (
        isinstance(defaults["tolerance"], bool)
        or not isinstance(defaults["tolerance"], (int, float))
        or not np.isfinite(defaults["tolerance"])
        or defaults["tolerance"] <= 0
    ):
        raise ValueError("refinement.tolerance must be finite and positive")
    return defaults
