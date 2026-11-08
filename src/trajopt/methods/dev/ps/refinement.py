"""Patterson--Hager--Rao refinement rule, doi:10.1002/oca.2114."""

import math
import numpy as np
from trajopt.methods.common.discretize import segment_ranges


def refine_mesh(mesh, errors, options):
    """Eqs. 34--35: raise the degree or split into equal smaller intervals."""
    ranges = segment_ranges(len(mesh.tau), mesh.segments)
    if (
        len(errors) != len(ranges)
        or not np.isfinite(errors).all()
        or np.any(np.asarray(errors) < 0)
    ):
        raise ValueError("Expected one finite nonnegative error per mesh interval")
    knots, degrees = [0.0], []
    for (a, b), error in zip(ranges, errors):
        degree = b - a
        if error <= options["tolerance"]:
            knots.append(float(mesh.tau[b]))
            degrees.append(degree)
            continue
        if degree < 2:
            predicted = options["min_degree"]
        else:
            predicted = degree + math.ceil(
                math.log(error / options["tolerance"]) / math.log(degree)
            )
        if predicted <= options["max_degree"]:
            knots.append(float(mesh.tau[b]))
            degrees.append(max(predicted, options["min_degree"]))
        else:
            count = max(math.ceil(predicted / options["min_degree"]), 2)
            knots.extend(np.linspace(mesh.tau[a], mesh.tau[b], count + 1)[1:].tolist())
            degrees.extend([options["min_degree"]] * count)
    return dict(knots=knots, degrees=degrees)
