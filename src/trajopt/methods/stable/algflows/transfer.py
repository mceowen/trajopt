"""Check physical channel compatibility before importing a stage's guesses."""


def transfer_guesses(source, target_layouts, phases=None):
    selected = tuple(source.trajectory) if phases is None else tuple(phases)
    if not selected:
        raise ValueError(f"Stage '{source.name}' has no trajectory to transfer")
    guesses = {}
    for name in selected:
        if name not in source.trajectory:
            raise ValueError(f"Source stage '{source.name}' has no phase '{name}'")
        if name not in target_layouts:
            raise ValueError(f"Destination has no phase '{name}'")
        if source.layouts[name] != target_layouts[name]:
            raise ValueError(f"Phase '{name}' has incompatible channel identities, ordering, or units")
        trajectory = source.trajectory[name]
        guesses[name] = trajectory.without_aux()
    return guesses
