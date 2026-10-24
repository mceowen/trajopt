"""Evaluate diagnostics on the accepted trajectory without changing solver state."""
import copy

import cvxpy as cp
import numpy as np

from trajopt.utils._snapshot import readonly_array


def _number(value):
    """Sum a finite numeric value or a constant CVXPY expression."""
    if isinstance(value, cp.Expression):
        if value.variables() or value.parameters():
            raise ValueError('Objective callback returned a live optimization expression')
        value = value.value
    if value is None:
        raise ValueError('Objective callback did not produce a value')
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError('Objective contribution is nonfinite')
    return float(np.sum(array))


def _evaluate_cost(segment, cost, z, nu, params):
    """Evaluate the configured SCP cost from the supplied trajectory arrays."""
    kind = cost.type
    idx = segment.index_map.indices
    if kind == 'nonconvex_running':
        gamma = getattr(cost, 'gamma_idx', None)
        if gamma is None or not len(idx.z.running_cost):
            return 0., 'No running-cost accumulator is installed; recipe adds no cost'
        return (_number(cost.w*z[-1, idx.z.running_cost[gamma]]),
                'Accepted optimized running-cost accumulator; no independent quadrature')
    if kind in {'nonconvex_terminal', 'nonconvex_minimax'}:
        nodes = np.atleast_1d(cost.nodes)
        values = np.array([np.atleast_1d(cost.fcn_znu(z[k], nu[k], params)) for k in nodes])
        if kind == 'nonconvex_minimax':
            if not len(nodes):
                raise ValueError('A minimax cost needs at least one evaluation node')
            return (_number(cost.w*np.max(values, axis=0)),
                    'Componentwise maximum of nonlinear callback values at configured accepted nodes')
        # The current terminal recipe does not apply a separate w factor.
        return (_number(values), 'Nonlinear callback at configured accepted nodes; configured output scale applied')
    if kind in {'convex_terminal', 'convex_running'}:
        # Match the implemented recipe: normalized x/u and literal time zero.
        # Constructing Constants allows CP-aware callbacks without using any
        # backend variable, reference parameter, or previous solve expression.
        nodes = np.atleast_1d(cost.nodes)
        node_weights = np.ones(len(z))
        if kind == 'convex_running':
            if segment.mesh.weights is not None:
                node_weights = np.concatenate(([0.], len(z)*segment.mesh.weights))
            nodes = [k for k in range(len(z)) if k in nodes]
        value = 0.
        for k in nodes:
            if node_weights[k] == 0:
                continue
            x = cp.Constant(z[k, idx.z.state])
            u = cp.Constant(nu[k, idx.nu.control])
            value += _number(node_weights[k]*cost.w*cost.fcn_dim(x, u, 0, params))
        return value, 'Configured nodal convex callback on normalized x/u with time zero, matching recipe'
    if kind == 'min_time':
        if not segment.free_final_time:
            return 0., 'Fixed-time recipe adds no minimum-time objective'
        from trajopt.methods.common.scp.cost_types import min_time_all_nodes, min_time_final_node
        recipe = segment.fcns.cost.min_time
        if recipe is min_time_final_node:
            return _number(z[-1, idx.z.time]), 'Final normalized time, matching min_time_final_node'
        if recipe is min_time_all_nodes:
            return _number(z[:, idx.z.time]), 'Sum of normalized nodal times, matching min_time_all_nodes'
        raise ValueError('Unrecognized minimum-time recipe')
    if kind == 'min_norm_terminal':
        target = cost.value if cost.value is not None else np.zeros(len(cost.idx))
        return _number(np.linalg.norm(z[-1, cost.idx]-target)), 'Terminal normalized-state Euclidean norm'
    if kind == 'final_state':
        return _number(cost.w*z[-1, cost.idx]), 'Weighted selected final normalized-state coordinates'
    if kind == 'final_control':
        return _number(cost.w*nu[-1, cost.idx]), 'Weighted selected final normalized-control coordinates'
    if kind in {'regularization', 'rate_regularization'}:
        if cost.set == 'state':
            values = z
        elif cost.set == 'control':
            values = nu
        else:
            raise ValueError(f'Unrecognized regularization set {cost.set!r}')
        if kind == 'rate_regularization':
            values = np.diff(values[:, cost.idx], axis=0)
            source = 'Adjacent normalized-coordinate differences on selected indices; no time division'
        else:
            source = 'All augmented normalized coordinates, matching the implemented regularization recipe'
        if cost.norm_type == 'l2':
            value = np.sum(values**2)
        elif cost.norm_type == 'l1':
            value = np.sum(np.abs(values))
        else:
            raise ValueError(f'Unrecognized regularization norm {cost.norm_type!r}')
        return _number(cost.w*value), source
    raise ValueError(f'Unsupported objective type {kind!r}')


def objective_terms(segment):
    """Evaluate the configured objective on the accepted reference trajectory.

    Terms use the configured scales and weights. Their physical units can
    differ. The objective excludes solver penalties. Running costs use the
    accepted cost accumulators without further integration.

    If any term cannot be evaluated, the total objective is unavailable.
    This function does not change the solver state.
    """
    z = readonly_array(segment.current_iter_data.z_opt)
    nu = readonly_array(segment.current_iter_data.nu_opt)
    params = copy.deepcopy(segment.params)
    terms, unsupported = {}, []
    for name, wrapper in segment.costs.items():
        cost = wrapper.cost
        term = {'type': cost.type, 'value': None,
                'units': 'configured normalized objective contribution'}
        if hasattr(cost, 'scale'):
            term['configured_scale'] = cost.scale
        try:
            term['value'], term['source'] = _evaluate_cost(segment, cost, z, nu, params)
        except Exception as exc:
            unsupported.append(name)
            term['source'] = 'unavailable'
            term['reason'] = f'{type(exc).__name__}: {exc}'
        terms[name] = term
    return {'objective': None if unsupported else float(sum(term['value'] for term in terms.values())),
            'objective_terms': terms,
            'objective_kind': 'configured scaled objective on accepted reference, excluding algorithm penalties and proximal costs',
            'unsupported_objective_terms': tuple(unsupported)}
