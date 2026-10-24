"""Select a joint step using the global SCP settings."""
from trajopt.methods.common.trust_region import line_search


def select_step(segments, method_spec):
    if not method_spec.line_search:
        return 1.0
    return line_search(segments, alpha_min=method_spec.alpha_min_ls)
