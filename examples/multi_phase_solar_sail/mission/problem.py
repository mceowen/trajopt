"""Build the alternating sunlit/eclipse phases using native problem inputs."""

from pathlib import Path
from importlib.resources import files
import numpy as np
from trajopt.common.tools import AttrDict, recursive_attrdict

_SOURCE = str(Path(__file__).with_name("callbacks.py").resolve())
_DYNAMICS_SOURCE = str(files("trajopt") / "models/solar_sail/solar_sail.py")
_STATE_NAMES = ("semi_latus_offset", "f", "g", "h", "k", "elapsed_time")
_MIN_ARC_RAD = 0.05


def shadow_drift_per_revolution(settings: AttrDict) -> float:
    """Longitude the shadow drifts per revolution, from the Sun's own motion."""
    rate = float(settings.sun_angular_rate_deg_per_day) * np.pi / 180.0
    period_days = (
        2.0
        * np.pi
        * np.sqrt(
            float(settings.constants.geo_radius_km) ** 3 / float(settings.constants.earth_mu_km3_s2)
        )
        / float(settings.constants.day_s)
    )
    return rate * period_days


def shadow_centre_longitude(settings: AttrDict, revolution: int) -> float:
    """Longitude of the ``revolution``-th shadow centre, Sun drift included."""
    return 2.0 * np.pi * revolution + shadow_drift_per_revolution(settings) * revolution


def shadow_entry_longitude(settings: AttrDict, revolution: int) -> float:
    """Longitude at which the ``revolution``-th eclipse begins."""
    return shadow_centre_longitude(settings, revolution) - half_shadow_angle(settings)


def final_longitude(settings: AttrDict) -> float:
    """Longitude at which the transfer ends: the last shadow entry."""
    return shadow_entry_longitude(settings, int(settings.revolutions))


def max_flown_revolutions(settings: AttrDict) -> float:
    """Most revolutions the trajectory can cover."""
    return (final_longitude(settings) - float(settings.initial_longitude_rad)) / (2.0 * np.pi)


def _state_box(settings: AttrDict) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper state bounds, in dimensional units (paper Eq. 44)."""
    bounds = settings.state_bounds
    constants = settings.constants
    semi_latus = np.asarray(bounds.semi_latus_offset_km, dtype=float)
    eccentricity = np.asarray(bounds.eccentricity_component, dtype=float)
    inclination = np.asarray(bounds.inclination_component, dtype=float)
    elapsed_upper = (
        float(bounds.days_per_revolution_bound)
        * max_flown_revolutions(settings)
        * float(constants.day_s)
    )
    lower = np.array(
        [semi_latus[0], eccentricity[0], eccentricity[0], inclination[0], inclination[0], 0.0]
    )
    upper = np.array(
        [
            semi_latus[1],
            eccentricity[1],
            eccentricity[1],
            inclination[1],
            inclination[1],
            elapsed_upper,
        ]
    )
    return (lower, upper)


def _state_scales(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Scaling factor for each state, from its allowed interval (paper Eq. 44)."""
    amplitude = upper - lower
    magnitude = np.abs(0.5 * (lower + upper))
    return np.where(lower * upper > 0.0, magnitude, amplitude)


def defect_tolerance(settings: AttrDict, phases: list) -> float:
    """Per-interval dynamics-defect tolerance, sized from an accumulated budget."""
    budget_km = float(settings.max_accumulated_defect_km)
    lower, upper = _state_box(settings)
    p_scale = _state_scales(lower, upper)[0]
    n_intervals = sum(
        (
            int(settings.sunlit_nodes if sunlit else settings.eclipse_nodes) - 1
            for _, _, sunlit in phases
        )
    )
    return budget_km / p_scale / n_intervals


def _initial_state(settings: AttrDict) -> np.ndarray:
    """Circular equatorial GEO: p = r_GEO, i.e. offset 0, and f = g = h = k = t = 0."""
    del settings
    return np.zeros(6)


def _steering_guess(
    progress: float, settings: AttrDict, longitude: float = 0.0, initial_longitude: float = 0.0
) -> float:
    """Steering angle for the initial guess."""
    start = float(settings.guess.steering_angle_start_rad)
    stop = float(settings.guess.steering_angle_stop_rad)
    base = (1.0 - progress) * start + progress * stop
    winds = float(settings.guess.get("steering_winds_per_rev", 1.0))
    return base - winds * (longitude - initial_longitude)


def half_shadow_angle(settings: AttrDict) -> float:
    """Half-angle subtended by the cylindrical Earth shadow at GEO."""
    return float(np.arcsin(settings.constants.earth_radius_km / settings.constants.geo_radius_km))


def phase_boundaries(settings: AttrDict) -> list[tuple[float, float, bool]]:
    """Nominal ``(longitude_start, longitude_stop, is_sunlit)`` for every arc."""
    n_rev = int(settings.revolutions)
    if n_rev < 1 or n_rev != settings.revolutions:
        raise ValueError("revolutions must be a positive integer")
    start = float(settings.initial_longitude_rad)
    stop = final_longitude(settings)
    if stop <= start + _MIN_ARC_RAD:
        raise ValueError("The final longitude must follow the initial longitude")
    half_shadow = half_shadow_angle(settings)
    crossings = []
    for revolution in range(1, n_rev + 1):
        centre = shadow_centre_longitude(settings, revolution)
        crossings.extend((centre - half_shadow, centre + half_shadow))
    entries = []
    for _ in range(1, n_rev + 1):
        entries.extend((True, False))
    inside = [
        (L, is_entry)
        for L, is_entry in zip(crossings, entries)
        if start + _MIN_ARC_RAD < L < stop - _MIN_ARC_RAD
    ]
    crossings = [L for L, _ in inside]
    sunlit_flags = [True]
    for _, is_entry in inside:
        sunlit_flags.append(not is_entry)
    edges = [start, *crossings, stop]
    return [
        (arc_start, arc_stop, sunlit)
        for arc_start, arc_stop, sunlit in zip(edges[:-1], edges[1:], sunlit_flags)
    ]


def _build_segment(settings: AttrDict, sunlit: bool) -> AttrDict:
    """Return the parts of a segment configuration that every arc shares."""
    constants = settings.constants
    lower, upper = _state_box(settings)
    scales = _state_scales(lower, upper)
    params = {
        **dict(constants),
        **dict(settings.sail),
        "sun_direction_eci": list(settings.sun_direction_eci),
        "sun_angular_rate_rad_s": float(
            np.deg2rad(settings.sun_angular_rate_deg_per_day) / constants.day_s
        ),
        "illumination": 1.0 if sunlit else 0.0,
        "final_eccentricity_max": float(
            settings.graveyard.eccentricity_max
            - float(settings.graveyard.get("eccentricity_margin", 0.0))
        ),
        "final_perigee_radius_km": float(
            constants.geo_radius_km
            + settings.graveyard.perigee_altitude_km
            + float(settings.graveyard.get("perigee_margin_km", 0.0))
        ),
    }
    return recursive_attrdict(
        {
            "num_nodes": int(settings.sunlit_nodes if sunlit else settings.eclipse_nodes),
            "hp_segments": int(
                settings.get("sunlit_hp_segments" if sunlit else "eclipse_hp_segments", 1)
            ),
            "fcns": {
                "dynamics": f"{_DYNAMICS_SOURCE}:dynamics",
                "shadow_boundary": f"{_SOURCE}:shadow_boundary",
                "phase_end_longitude": f"{_SOURCE}:phase_end_longitude",
                "terminal_eccentricity_cone": f"{_SOURCE}:terminal_eccentricity_cone",
                "terminal_perigee_cone": f"{_SOURCE}:terminal_perigee_cone",
                "terminal_eccentricity": f"{_SOURCE}:terminal_eccentricity",
                "terminal_perigee_radius": f"{_SOURCE}:terminal_perigee_radius",
            },
            "params": params,
            "state": {
                name: {"idx": [index], "scale": float(scales[index])}
                for index, name in enumerate(_STATE_NAMES)
            },
            "control": {"steering_angle": {"idx": [0], "scale": 1.0}},
            "time": {"scale": 2.0 * np.pi * float(settings.revolutions)},
            "constraints": {
                "dynamics": {"type": "dynamics", "fcn": "fcns.dynamics"},
                "state_limits": {
                    "type": "state_limits",
                    "lower": lower.tolist(),
                    "upper": upper.tolist(),
                },
                "steering_limits": {
                    "type": "control_limits",
                    "lower": [-2.0 * np.pi * (settings.revolutions + 2)],
                    "upper": [2.0 * np.pi * (settings.revolutions + 2)],
                },
                "steering_rate_limit": {
                    "type": "control_rate_limit",
                    "value": [float(settings.max_steering_rate_rad_per_rad)],
                },
            },
            "costs": {},
            "outputs": {},
            "guess": {"type": "propagation"},
        }
    )


def _add_minimum_time_cost(segment: AttrDict, settings: AttrDict) -> None:
    """Attach the minimum-time objective to the final segment."""
    form = str(settings.minimum_time_objective)
    weight = float(settings.elapsed_time_cost_weight)
    if form in ("longitude", "both"):
        segment.costs.minimum_longitude = recursive_attrdict({"type": "min_time"})
    if form in ("elapsed_time", "both"):
        segment.costs.minimum_elapsed_time = recursive_attrdict(
            {"type": "final_state", "idx": 5, "w": 1.0 if form == "elapsed_time" else weight}
        )
    if form not in ("longitude", "elapsed_time", "both"):
        raise ValueError(
            f"minimum_time_objective must be 'longitude', 'elapsed_time' or 'both', got '{form}'"
        )


def build_trajectory_config(settings: AttrDict) -> AttrDict:
    """Expand the compact problem settings into native phase inputs."""
    missing = [key for key in ("sunlit_nodes", "eclipse_nodes") if key not in settings]
    if missing:
        raise ValueError(f"Set problem.{' and problem.'.join(missing)} in the example config")
    for key in ("sunlit_nodes", "eclipse_nodes"):
        if int(settings[key]) != settings[key] or settings[key] < 2:
            raise ValueError(f"{key} must be an integer of at least 2")
    if settings.max_accumulated_defect_km <= 0:
        raise ValueError("max_accumulated_defect_km must be positive")
    phases = phase_boundaries(settings)
    x0 = _initial_state(settings)
    initial_longitude = phases[0][0]
    total_longitude = phases[-1][1] - initial_longitude
    window = float(settings.terminator_window_rad)
    freedom = float(settings.final_longitude_freedom_rad)
    eps_dynamics = defect_tolerance(settings, phases)
    segments = AttrDict()
    previous_name = None
    for index, (arc_start, arc_stop, sunlit) in enumerate(phases, start=1):
        name = f"phase_{index:03d}_{('sunlit' if sunlit else 'eclipse')}"
        segment = _build_segment(settings, sunlit)
        segment.constraints.dynamics.eps = eps_dynamics
        is_last = index == len(phases)
        progress_start = (arc_start - initial_longitude) / total_longitude
        progress_stop = (arc_stop - initial_longitude) / total_longitude
        segment.guess.update(
            {
                "t_start": float(arc_start),
                "t_stop": float(arc_stop),
                "x_start": x0.tolist() if previous_name is None else "previous",
                "u_start": [
                    _steering_guess(progress_start, settings, arc_start, initial_longitude)
                ],
                "u_stop": [_steering_guess(progress_stop, settings, arc_stop, initial_longitude)],
            }
        )
        if previous_name is None:
            segment.constraints.initial_state = recursive_attrdict(
                {"type": "initial_state", "value": x0.tolist()}
            )
            segment.constraints.initial_longitude = recursive_attrdict(
                {"type": "initial_time", "value": float(arc_start)}
            )
        else:
            segment.constraints.state_continuity = recursive_attrdict(
                {"type": "state_continuity", "phase": previous_name}
            )
            segment.constraints.longitude_continuity = recursive_attrdict(
                {"type": "time_continuity", "phase": previous_name}
            )
            segment.constraints.control_continuity = recursive_attrdict(
                {"type": "control_continuity", "phase": previous_name}
            )
        if is_last:
            end_lower = max(arc_start + _MIN_ARC_RAD, arc_stop - freedom)
            end_upper = arc_stop
        else:
            if bool(settings.enforce_shadow_terminator):
                segment.constraints.ends_on_terminator = recursive_attrdict(
                    {
                        "type": "final_nonconvex_equality",
                        "fcn": "fcns.shadow_boundary",
                        "scale": [1.0],
                    }
                )
            end_lower, end_upper = (arc_stop - window, arc_stop + window)
        segment.constraints.arc_end_window = recursive_attrdict(
            {
                "type": "final_nonconvex_inequality",
                "fcn": "fcns.phase_end_longitude",
                "lower": np.array([float(end_lower)]),
                "upper": np.array([float(end_upper)]),
                "scale": [2.0 * np.pi],
            }
        )
        if is_last:
            if settings.graveyard.get("buffer_terminal", False):
                params = {
                    "eccentricity": float(
                        settings.graveyard.eccentricity_max
                        - float(settings.graveyard.get("eccentricity_margin", 0.0))
                    ),
                    "perigee": float(
                        settings.constants.geo_radius_km
                        + settings.graveyard.perigee_altitude_km
                        + float(settings.graveyard.get("perigee_margin_km", 0.0))
                    ),
                }
                segment.constraints.final_eccentricity = recursive_attrdict(
                    {
                        "type": "final_nonconvex_inequality",
                        "fcn": "fcns.terminal_eccentricity",
                        "upper": np.array([params["eccentricity"]]),
                        "scale": [0.02],
                    }
                )
                segment.constraints.final_perigee = recursive_attrdict(
                    {
                        "type": "final_nonconvex_inequality",
                        "fcn": "fcns.terminal_perigee_radius",
                        "lower": np.array([params["perigee"]]),
                        "scale": [1000.0],
                    }
                )
            else:
                segment.constraints.final_eccentricity = recursive_attrdict(
                    {"type": "final_convex_inequality", "fcn": "fcns.terminal_eccentricity_cone"}
                )
                segment.constraints.final_perigee = recursive_attrdict(
                    {"type": "final_convex_inequality", "fcn": "fcns.terminal_perigee_cone"}
                )
            _add_minimum_time_cost(segment, settings)
        segments[name] = segment
        previous_name = name
    return recursive_attrdict({"phases": segments})
