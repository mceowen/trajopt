"""Replay native controls through the full phase chain from circular GEO."""

import numpy as np

from trajopt.analysis.validation import detached_phases
from trajopt.methods.common.discretize import propagate_trajectory


def validate(result, settings, *, step_multiplier=1):
    """Check the terminal orbit and defect budget without terminal slack.

    These checks do not certify continuous-time path feasibility.
    """
    if not result.usable:
        return {"solver_converged": False, "accepted": False, "reason": result.reason}, {}
    if not isinstance(step_multiplier, int) or step_multiplier < 1:
        raise ValueError("step_multiplier must be a positive integer")
    dense = {name: [] for name in ("state", "longitude", "control", "illuminated")}
    end = previous = None
    max_defect = state_jump = longitude_jump = 0.0
    min_duration = float("inf")
    intervals = 0
    for adapter in detached_phases(result).values():
        phase = adapter.problem
        idx = phase.index_map.indices
        data = adapter.iter_data_list[-1]
        z, nu = np.array(data.z_opt), np.array(data.nu_opt)
        scales = np.asarray(phase.nondim.state_scales)
        if previous is not None:
            state_jump = max(
                state_jump, float(np.max(np.abs(z[0, idx.z.state] - previous[idx.z.state])))
            )
            longitude_jump = max(
                longitude_jump,
                float(np.max(np.abs(z[0, idx.z.time] - previous[idx.z.time])))
                * phase.nondim.time_scale,
            )
        previous = z[-1].copy()
        claimed = z[-1, idx.z.state] * scales
        min_duration = min(
            min_duration, float((z[-1, idx.z.state[5]] - z[0, idx.z.state[5]]) * scales[5])
        )
        max_defect = max(max_defect, float(np.max(np.abs(data.get("defect", 0.0)))))
        intervals += len(z) - 1
        if end is None:
            z[0, idx.z.state] = 0.0
            z[0, idx.z.time] = settings.initial_longitude_rad / phase.nondim.time_scale
        else:
            z[0] = end
        dynamics = next(c for c in phase.constraints.values() if c.type == "dynamics")
        _, states, controls = propagate_trajectory(
            z,
            adapter.mesh.tau,
            nu,
            dynamics.fcn_znu,
            phase.params,
            discretize=adapter.mesh.discretize,
            segments=adapter.mesh.segments,
            n_steps=step_multiplier * max(200, 40 * (len(z) - 1)),
            dilation_idx=idx.nu.dilation_factor,
        )
        end = states[-1]
        keep = slice(None) if not dense["state"] else slice(1, None)
        dense["state"].append(states[keep][:, idx.z.state] * scales)
        dense["longitude"].append(states[keep][:, idx.z.time].ravel() * phase.nondim.time_scale)
        dense["control"].append(controls[keep][:, idx.nu.control] * phase.nondim.control_scales)
        dense["illuminated"].append(np.full(len(states[keep]), bool(phase.params.illumination)))
    dense = {key: np.concatenate(value) for key, value in dense.items()}
    true = dense["state"][-1]
    geo = float(settings.constants.geo_radius_km)
    eccentricity = float(np.hypot(true[1], true[2]))
    perigee = float((geo + true[0]) / (1.0 + eccentricity) - geo)
    defect = float(claimed[0] - true[0])
    eps = float(settings.max_accumulated_defect_km / scales[0] / intervals)
    terminal_met = (
        perigee >= settings.graveyard.perigee_altitude_km
        and eccentricity <= settings.graveyard.eccentricity_max
    )
    report = dict(
        solver_converged=result.converged,
        iterations=result.iterations,
        reason=result.reason,
        revolutions=int(settings.revolutions),
        nodal_transfer_days=float(claimed[5] / settings.constants.day_s),
        replay_transfer_days=float(true[5] / settings.constants.day_s),
        replay_perigee_above_geo_km=perigee,
        replay_eccentricity=eccentricity,
        terminal_requirements_met=bool(terminal_met),
        accumulated_semilatus_defect_km=defect,
        within_defect_budget=abs(defect) <= settings.max_accumulated_defect_km,
        max_dynamics_defect_nd=max_defect,
        dynamics_tolerance_nd=eps,
        max_state_jump_nd=state_jump,
        max_longitude_jump_rad=longitude_jump,
        shortest_phase_s=min_duration,
        replay_step_multiplier=step_multiplier,
    )
    report["accepted"] = bool(
        result.converged
        and terminal_met
        and report["within_defect_budget"]
        and max_defect <= eps
        and state_jump <= 1e-4
        and longitude_jump <= 1e-8
        and min_duration > 0
    )
    return report, dense
