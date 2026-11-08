"""Check physical channel compatibility before importing a stage's guesses."""

import copy

import numpy as np

from trajopt.methods.common.penalties import install_penalty_state


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


def capture_endpoint_weights(method):
    """Copy the final penalty weights and the endpoint constraints they apply to."""
    states = {}
    for name, phase in method.phases.items():
        states[name] = {}
        for cname, constraint in phase.constraints.items():
            if constraint.type != 'final_state':
                continue
            physical, penalty = constraint.constraint, constraint.penalties
            idx = np.asarray(physical.idx, dtype=int)
            weight = penalty.get('W')
            states[name][cname] = dict(
                idx=idx.copy(), value=np.array(physical.value, copy=True),
                scaling=phase.nondim.M.state.d2nd[np.ix_(idx, idx)].copy(),
                norm=penalty.get('norm'), vb_type=penalty.vb_type,
                W=None if weight is None else weight.copy())
    return states


def transfer_endpoint_weights(source, method, phases):
    """Continue matching endpoint weights without transferring duals or tolerances."""
    targets = capture_endpoint_weights(method)
    pending = []
    for name in phases:
        old, new = source.endpoint_weights.get(name, {}), targets[name]
        if old.keys() != new.keys():
            raise ValueError(f"Phase '{name}' has different endpoint constraints")
        for cname, target in new.items():
            origin = old[cname]
            compatible = (
                origin['vb_type'] == target['vb_type'] == 'standard'
                and origin['norm'] == target['norm']
                and all(np.array_equal(origin[key], target[key])
                        for key in ('idx', 'value', 'scaling'))
                and origin['W'] is not None and target['W'] is not None
                and origin['W'].shape == target['W'].shape
                and np.isfinite(origin['W']).all() and (origin['W'] >= 0).all())
            if not compatible:
                raise ValueError(f"Phase '{name}' endpoint '{cname}' has incompatible penalty weights")
            pending.append((method.phases[name].constraints[cname].penalties, origin['W']))
    for penalty, weight in pending:
        install_penalty_state(penalty, {'W': weight})
    for name in phases:
        phase = method.phases[name]
        phase.current_iter_data.update(phase.penalty_snapshot())
        phase.iter_data_list[0].update(copy.deepcopy(phase.penalty_snapshot()))
