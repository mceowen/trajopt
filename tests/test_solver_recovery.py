"""Recovery must retry without applying an invalid candidate to the accepted iterate."""
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import Mock

import cvxpy as cp
import numpy as np
import pytest

from trajopt.methods.common import trust_region
from trajopt.methods.common.scp.subproblem import SCPSubproblem
from trajopt.methods.dev.scvx.scp_method import SCPMethod as SimpleSCP
from trajopt.methods.dev.scvx_phases.scp_method import SCPMethod as PhaseSCP
from trajopt.methods.dev.sqp.sqp_method import SQPMethod as SimpleSQP
from trajopt.methods.dev.sqp_phases.sqp_method import SQPMethod as PhaseSQP
from trajopt.utils.tools import recursive_attrdict

METHODS = [SimpleSCP, PhaseSCP, SimpleSQP, PhaseSQP]


def make_method(method_class, outcomes, *, iter_max=3, lm_mu=0.0):
    """Use the real solve loop and validity check with a controlled solver."""
    method = method_class.__new__(method_class)
    method.method_config = recursive_attrdict({'flags': {'iter_max': iter_max}, 'solver_opts': {}})
    method._converged = False
    phased = method_class in (PhaseSCP, PhaseSQP)
    blocks = [NS(
        name=f'block{i}', tr_scale=1.0, lm_mu=lm_mu,
        dz=NS(value=np.ones((1, 1))), dnu=NS(value=np.ones((1, 1))),
        current_iter_data=NS(discretization_time=0, t_start=0, t_final=1),
        iter_data_list=[NS(iter_num=0)],
    ) for i in range(2 if phased else 1)]
    method.subproblem = blocks[0]
    method.scp_trajectory = NS(scp_phases={b.name: b for b in blocks})
    method.reporter = Mock()
    method.warmup_jax = Mock()
    method.update_cvxpy_parameters = Mock()
    method.display_status = Mock()
    attempts, accepted = [], []
    sqp = method_class in (SimpleSQP, PhaseSQP)

    def solve(**kwargs):
        # Until a valid candidate is accepted, the trajectory/history remain untouched.
        assert not accepted
        assert all(len(b.iter_data_list) == 1 for b in blocks)
        attempts.append([b.lm_mu if sqp else b.tr_scale for b in blocks])
        outcome = outcomes[min(len(attempts) - 1, len(outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        method.cp_subproblem.status = 'optimal' if outcome == 'valid' else 'user_limit'
        for b in blocks:
            b.dz.value = np.ones((1, 1))
            b.dnu.value = np.ones((1, 1))
        # In multiphase, only the last block is invalid: the whole candidate must wait.
        if outcome != 'valid':
            field, value = outcome
            getattr(blocks[-1], field).value = value

    def accept():
        assert trust_region.step_is_usable(blocks)
        accepted.append([b.lm_mu if sqp else b.tr_scale for b in blocks])
        for b in blocks:
            b.iter_data_list.append(NS(iter_num=1))
        method._converged = True

    method.cp_subproblem = NS(solve=solve, status='optimal', solver_stats=NS(solve_time=0))
    method.update_current_iter_data = Mock(side_effect=accept)
    return method, blocks, attempts, accepted


@pytest.mark.parametrize('method_class', METHODS)
def test_transient_solver_error_retries_without_accepting(method_class):
    method, blocks, attempts, accepted = make_method(
        method_class, [cp.error.SolverError('transient refusal'), 'valid'])
    method.solve()
    expected = 1e-5 if method_class in (SimpleSQP, PhaseSQP) else 10.0
    assert len(attempts) == 2
    assert attempts[1] == pytest.approx([expected] * len(blocks))
    assert len(accepted) == 1
    if method_class in (SimpleSCP, PhaseSCP):
        assert accepted[0] == [7.0] * len(blocks)
    assert method._converged
    method.update_current_iter_data.assert_called_once()


@pytest.mark.parametrize('method_class', METHODS)
@pytest.mark.parametrize('field', ['dz', 'dnu'])
@pytest.mark.parametrize('value', [None, np.array([[np.nan]]), np.array([[np.inf]]), np.array([[1e7]])])
def test_invalid_candidate_retries_before_application(method_class, field, value):
    method, blocks, attempts, accepted = make_method(
        method_class, [(field, value), 'valid'], lm_mu=1e-3)
    method.solve()
    expected = 1e-2 if method_class in (SimpleSQP, PhaseSQP) else 10.0
    assert attempts[1] == pytest.approx([expected] * len(blocks))
    assert len(accepted) == 1
    method.update_current_iter_data.assert_called_once()


@pytest.mark.parametrize('method_class', METHODS)
def test_persistent_refusal_respects_iteration_budget(method_class):
    method, blocks, attempts, accepted = make_method(
        method_class, [cp.error.SolverError('persistent refusal')], iter_max=2)
    method.solve()
    assert len(attempts) == 3  # Preserve the existing inclusive iter_max convention.
    assert not accepted
    assert not method._converged
    assert all(len(b.iter_data_list) == 1 for b in blocks)
    method.update_current_iter_data.assert_not_called()


@pytest.mark.parametrize('method_class', METHODS)
def test_valid_candidate_does_not_retry(method_class):
    method, blocks, attempts, accepted = make_method(method_class, ['valid'])
    method.solve()
    assert len(attempts) == len(accepted) == 1
    if method_class in (SimpleSCP, PhaseSCP):
        assert accepted[0] == [1.0] * len(blocks)


def test_scp_recovery_scale_is_bounded_and_relaxes_to_base():
    blocks = [NS(tr_scale=1.0), NS(tr_scale=1.0e6)]
    for _ in range(10):
        trust_region.tighten_scp_trust_region(blocks)
    assert [b.tr_scale for b in blocks] == [1.0e6, 1.0e6]
    for _ in range(50):
        trust_region.relax_scp_trust_region(blocks)
    assert [b.tr_scale for b in blocks] == [1.0, 1.0]


@pytest.mark.parametrize('augmented', [False, True])
@pytest.mark.parametrize('step_config', [
    {'z': 2.0, 'nu': 4.0},
    {'z': {'default': 2.0, 'x': 4.0, 'gamma': 8.0}, 'nu': {'default': 4.0, 's': 8.0}},
])
def test_recovery_refreshes_all_weights_without_mutating_config(augmented, step_config):
    block = SCPSubproblem.__new__(SCPSubproblem)
    block.penalty_config = recursive_attrdict({'tr_step': deepcopy(step_config)})
    original_config = deepcopy(block.penalty_config)
    block.tr_scale = 10.0
    block.flags = NS(discretize='ms')
    block.index_map = NS(
        n=NS(ctcs=int(augmented), running_cost=int(augmented)),
        unpack_znu=Mock(return_value=(None,) * 5),
    )
    block.current_iter_data = NS(z_opt=np.ones((1, 1)), nu_opt=np.ones((1, 1)))
    weights = {'x': ('z', 2.0), 't': ('z', 2.0), 'u': ('nu', 4.0), 's': ('nu', 4.0)}
    if augmented:
        weights.update(ctcs=('z', 2.0), gamma=('z', 2.0))
    block.cp_params = recursive_attrdict({
        'z_ref': cp.Parameter((1, 1)), 'nu_ref': cp.Parameter((1, 1)),
        **{'tr_' + key: cp.Parameter(nonneg=True) for key in weights},
    })
    block.constraints, block.costs = {}, {}
    for scale in (10.0, 7.0, 1.0):
        block.tr_scale = scale
        block.update_cvxpy_parameters()
        for key, (group, default) in weights.items():
            raw = step_config[group]
            step = raw.get(key, raw.get('default', default)) if isinstance(raw, dict) else raw
            assert block.cp_params['tr_' + key].value == pytest.approx(scale / step)
    assert block.penalty_config == original_config
