from trajopt.analysis.experiments import _build_solver

HP_SEGMENTS = 7

def hp_segments_for(name):
    """Number of hp intervals for a variant name, or None for single-domain transcriptions."""
    return HP_SEGMENTS if name.lower().startswith("hp") else None

def with_trajectory(**updates):
    """Solver factory that edits trajectory settings (e.g. hp_segments) before building the method."""
    def factory(config):
        for key, value in updates.items():
            config.trajectory[key] = value
        return _build_solver(config)
    return factory
