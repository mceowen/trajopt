"""Build the alternating sunlit/eclipse segments for the GEO disposal problem.

Problem structure (Kelly and Bevilacqua 2021, Section 5.1)
----------------------------------------------------------
State   x = [p, f, g, h, k, t_elapsed]   (MEOE plus physical time)
Control u = [theta]                      (primer steering angle, rad)
Independent variable: true longitude L (rad), not physical time.

Each revolution crosses the Earth shadow once, so the trajectory is a chain of
alternating sunlit and eclipse arcs.  Every arc is one trajopt segment; the arcs
are tied together by state, control and time continuity.

The role of ``revolutions``
---------------------------
``revolutions`` is a *structural guess*, not a boundary condition.  It fixes how
many phases the problem has and it seeds the initial guess, but no arc endpoint
is held at a constant longitude:

  * every interior arc ends where the spacecraft actually crosses the shadow
    terminator, imposed as the nonconvex equality ``shadow_boundary = 0``, so the
    boundaries move as the orbit evolves;
  * the final arc ends wherever the minimum-time solution wants it to, within
    roughly one revolution either side of the nominal endpoint.

The right value of ``revolutions`` is therefore found by re-solving with one
revolution fewer until the problem stops being feasible, which is the outer loop
of the paper's Algorithm 1.  See :mod:`search`.

A window of ``terminator_window_rad`` around each nominal boundary keeps an arc
from collapsing or inverting.  The nominal eclipse arc is about 0.304 rad, so a
window of 0.05 rad still leaves every arc at least 0.2 rad long.

Simplifications relative to the reference
-----------------------------------------
* Only Earth gravity (no lunar/solar third-body perturbations).
* Sun line rotating uniformly in the equatorial plane (no ephemeris).

Scaling
-------
Every state is scaled by the amplitude of its allowed interval (paper Eq. 44),
and the longitude by the nominal swept longitude, so all nondimensional
variables and their derivatives are O(1).
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from trajopt.flows.transfer import phase_layouts, transfer_guesses
from trajopt.utils import config_loader
from trajopt.utils import tools
from trajopt.utils.tools import AttrDict, recursive_attrdict

_SOURCE = str(Path(__file__).with_name("solar_sail.py").resolve())
# The first state is p - r_GEO, not p: see solar_sail.py's module docstring.
_STATE_NAMES = ("semi_latus_offset", "f", "g", "h", "k", "elapsed_time")

# Shortest arc we ever allow, as a guard against degenerate phases.
_MIN_ARC_RAD = 0.05


def shadow_drift_per_revolution(settings: AttrDict) -> float:
    """Longitude the shadow drifts per revolution, from the Sun's own motion."""
    rate = float(settings.sun_angular_rate_deg_per_day) * np.pi / 180.0
    period_days = 2.0 * np.pi * np.sqrt(
        float(settings.constants.geo_radius_km) ** 3
        / float(settings.constants.earth_mu_km3_s2)
    ) / float(settings.constants.day_s)
    return rate * period_days


def shadow_centre_longitude(settings: AttrDict, revolution: int) -> float:
    """Longitude of the ``revolution``-th shadow centre, Sun drift included."""
    return (2.0 * np.pi * revolution
            + shadow_drift_per_revolution(settings) * revolution)


def shadow_entry_longitude(settings: AttrDict, revolution: int) -> float:
    """Longitude at which the ``revolution``-th eclipse begins."""
    return shadow_centre_longitude(settings, revolution) - half_shadow_angle(settings)


def final_longitude(settings: AttrDict) -> float:
    """Longitude at which the transfer ends: the last shadow entry.

    A sail transfer has no reason to continue into an eclipse -- thrust is zero
    there, so the orbit is unchanged across it -- and every longitude past that
    entry would need another eclipse phase to model it.  Ending the chain here
    is what keeps the last arc a full sunlit arc instead of a stub trailing a
    final eclipse that does no work.
    """
    return shadow_entry_longitude(settings, int(settings.revolutions))


def max_flown_revolutions(settings: AttrDict) -> float:
    """Most revolutions the trajectory can cover.

    The final arc may end early, never late, so this is exact rather than a
    bound: the transfer stops at the last shadow entry at the latest.
    """
    return (final_longitude(settings)
            - float(settings.initial_longitude_rad)) / (2.0 * np.pi)


def _state_box(settings: AttrDict) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper state bounds, in dimensional units (paper Eq. 44).

    The orbital-element bounds are set by the terminal orbit, which does not
    depend on the sail, so the same numbers serve any area-to-mass ratio.  The
    elapsed-time bound is the one exception: it has to grow with the revolution
    count, so it is derived from it rather than written out.  That keeps the
    revolution count the only thing that changes between sail sizings, and keeps
    the time state nondimensionalised to roughly [0, 1] in every case.
    """
    bounds = settings.state_bounds
    constants = settings.constants

    # already an offset from r_GEO, which is exactly what the state carries
    semi_latus = np.asarray(bounds.semi_latus_offset_km, dtype=float)
    eccentricity = np.asarray(bounds.eccentricity_component, dtype=float)
    inclination = np.asarray(bounds.inclination_component, dtype=float)
    elapsed_upper = (
        float(bounds.days_per_revolution_bound)
        * max_flown_revolutions(settings)
        * float(constants.day_s)
    )

    lower = np.array([semi_latus[0], eccentricity[0], eccentricity[0],
                      inclination[0], inclination[0], 0.0])
    upper = np.array([semi_latus[1], eccentricity[1], eccentricity[1],
                      inclination[1], inclination[1], elapsed_upper])
    return lower, upper


def _state_scales(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Scaling factor for each state, from its allowed interval (paper Eq. 44).

    The amplitude of the interval is the right scale only for a state that is
    centred near zero, because trajopt nondimensionalises by division alone
    with no offset.  Every state here is written so that it is: the semi-latus
    rectum is carried as an offset from r_GEO rather than as p itself (see
    solar_sail.py), so its interval straddles zero like the others and the
    amplitude is the correct scale throughout.

    The magnitude branch below is kept for any state added later whose interval
    does not contain zero, but nothing in this problem takes it.  Prefer
    re-centring such a state over relying on it: a scale that does not resolve
    a state's motion makes its dynamics defect numerically invisible, and the
    optimiser will spend that defect instead of thrusting.
    """
    amplitude = upper - lower
    magnitude = np.abs(0.5 * (lower + upper))
    return np.where(lower * upper > 0.0, magnitude, amplitude)


def defect_tolerance(settings: AttrDict, phases: list) -> float:
    """Per-interval dynamics-defect tolerance, sized from an accumulated budget.

    The solver bounds the defect of each *interval* -- each adjacent pair of
    nodes -- not of each phase.  Those errors accumulate along the trajectory,
    so what matters physically is the sum over every interval in the problem:

        accumulated ~= eps * p_scale * n_intervals

    with ``n_intervals = sum over phases of (num_nodes - 1)``.  Inverting that
    for a chosen accumulated budget gives the tolerance below.

    Sizing this matters here more than in most problems.  The semi-latus rectum
    is scaled by its magnitude (~42164 km, see _state_scales), so the default
    1e-4 tolerance is worth 4.2 km on *every interval*; over the 173 intervals
    of a three-revolution transfer that permits far more orbit raise than the
    sail itself delivers (~9 km/day), and the optimiser will spend it.  The
    budget has to be small against the real effect, not against the state.
    """
    budget_km = float(settings.max_accumulated_defect_km)
    lower, upper = _state_box(settings)
    p_scale = _state_scales(lower, upper)[0]

    n_intervals = sum(
        int(settings.sunlit_nodes if sunlit else settings.eclipse_nodes) - 1
        for _, _, sunlit in phases
    )
    return budget_km / p_scale / n_intervals


def _initial_state(settings: AttrDict) -> np.ndarray:
    """Circular equatorial GEO: p = r_GEO, i.e. offset 0, and f = g = h = k = t = 0."""
    del settings
    return np.zeros(6)


def _steering_guess(progress: float, settings: AttrDict,
                    longitude: float = 0.0, initial_longitude: float = 0.0) -> float:
    """Steering angle for the initial guess.

    The base value ramps between the configured endpoint primers.  On top of
    that the angle *winds*: theta is measured in the rotating local frame, so a
    primer that is fixed in inertial space -- which is what the optimal solution
    looks like, the sail being pushed away from a nearly stationary Sun -- turns
    by -2*pi per revolution in this frame.  Kelly & Bevilacqua Figure 7 shows
    exactly that: the primer angle sweeps the full circle once per orbit.

    Without the winding term the guess is a constant angle, and SCP is a local
    method: it stays near the non-winding profile, the sail cannot follow the
    optimal direction for more than about one revolution, and the eccentricity
    is pumped instead of being held at the cap.  ``guess.steering_winds_per_rev``
    scales the term; 0 recovers the old constant-angle guess.
    """
    start = float(settings.guess.steering_angle_start_rad)
    stop = float(settings.guess.steering_angle_stop_rad)
    base = (1.0 - progress) * start + progress * stop
    winds = float(settings.guess.get("steering_winds_per_rev", 1.0))
    return base - winds * (longitude - initial_longitude)


def half_shadow_angle(settings: AttrDict) -> float:
    """Half-angle subtended by the cylindrical Earth shadow at GEO."""
    return float(
        np.arcsin(settings.constants.earth_radius_km / settings.constants.geo_radius_km)
    )


def phase_boundaries(settings: AttrDict) -> list[tuple[float, float, bool]]:
    """Nominal ``(longitude_start, longitude_stop, is_sunlit)`` for every arc.

    These are the *structural* boundaries: they set the number of phases and
    seed the initial guess.  The solver is free to move them (see the module
    docstring).  The Sun is fixed along ``-sun_direction_eci``, so the shadow is
    centred on integer multiples of 2*pi and starting at L0 = pi puts the
    spacecraft in full sunlight at the initial time, as in the reference.
    """
    n_rev = int(settings.revolutions)
    if n_rev < 1:
        raise ValueError(f"revolutions must be at least 1, got {n_rev}")

    start = float(settings.initial_longitude_rad)
    # The transfer ends at the last shadow entry, not at a whole number of
    # revolutions past the start.  Running to start + 2*pi*n_rev instead puts
    # that entry *inside* the span, so the chain picks up one more eclipse and
    # then a final sunlit stub behind it -- and the minimum-time solution ends
    # in that stub, having gained nothing from the eclipse it was forced to fly
    # through.  See final_longitude().
    stop = final_longitude(settings)
    half_shadow = half_shadow_angle(settings)

    # The shadow is centred on the anti-Sun direction, so once the Sun line
    # moves the eclipse moves with it.  One revolution takes about one orbital
    # period, which gives the elapsed time at each crossing to the accuracy a
    # structural guess needs; the shadow_boundary equality then places each
    # boundary exactly.  Without this the nominal boundaries stay at 2*pi*i
    # while the true ones drift ~1 deg per day, and after a few revolutions they
    # fall outside terminator_window_rad and the guess is wrong by more than the
    # window can absorb.
    crossings = []
    for revolution in range(1, n_rev + 1):
        centre = shadow_centre_longitude(settings, revolution)
        crossings.extend((centre - half_shadow, centre + half_shadow))
    # Track which crossings are shadow *entries*, so each arc can be labelled by
    # construction.  Deciding it afterwards from the geometry -- "is the arc
    # midpoint within half_shadow of a multiple of 2*pi" -- silently assumes a
    # fixed Sun.  With the Sun moving, the crossings drift by Omega*t, and once
    # that exceeds half_shadow (about nine days here) every later eclipse arc is
    # mislabelled as sunlit and the sail is given thrust it should not have.
    entries = []
    for _ in range(1, n_rev + 1):
        entries.extend((True, False))  # (centre - half_shadow, centre + half_shadow)

    inside = [(L, is_entry) for L, is_entry in zip(crossings, entries)
              if start + _MIN_ARC_RAD < L < stop - _MIN_ARC_RAD]
    crossings = [L for L, _ in inside]

    # An arc is eclipsed when it begins at a shadow entry.
    sunlit_flags = [True]  # the trajectory starts in sunlight at L0 = pi
    for _, is_entry in inside:
        sunlit_flags.append(not is_entry)

    edges = [start, *crossings, stop]
    return [(arc_start, arc_stop, sunlit)
            for arc_start, arc_stop, sunlit
            in zip(edges[:-1], edges[1:], sunlit_flags)]


def _build_segment(settings: AttrDict, sunlit: bool) -> AttrDict:
    """Return the parts of a segment configuration that every arc shares."""
    constants = settings.constants
    lower, upper = _state_box(settings)
    scales = _state_scales(lower, upper)

    params = {
        **dict(constants),
        **dict(settings.sail),
        "sun_direction_eci": list(settings.sun_direction_eci),
        # Earth's orbital motion, as an angular rate of the Sun line about +Z.
        # Zero recovers the fixed-Sun model.
        "sun_angular_rate_rad_s": float(
            np.deg2rad(settings.sun_angular_rate_deg_per_day) / constants.day_s
        ),
        # 1 in sunlight, 0 in eclipse; multiplies the whole sail acceleration.
        "illumination": 1.0 if sunlit else 0.0,
        # The eccentricity cone gets the same treatment as the perigee cone
        # below, and for the same reason: minimum time parks the nodal value on
        # the cap, and the propagated one then sits over it by the interpolation
        # error.  Again the margin is applied here only.
        "final_eccentricity_max": float(
            settings.graveyard.eccentricity_max
            - float(settings.graveyard.get("eccentricity_margin", 0.0))
        ),
        # The cone is asked for the requirement PLUS a margin.  Minimum time
        # drives the *nodal* perigee exactly onto whatever it is given, and the
        # propagated trajectory then lands short by the discretisation error, so
        # aiming exactly at the requirement produces a solution that misses it.
        # The margin is added here only: solution.validation_report judges the
        # propagated orbit against the unmargined requirement, so this buys no
        # credit -- it only stops the optimiser stopping short.
        "final_perigee_radius_km": float(
            constants.geo_radius_km
            + settings.graveyard.perigee_altitude_km
            + float(settings.graveyard.get("perigee_margin_km", 0.0))
        ),
    }

    return recursive_attrdict({
        "num_nodes": int(settings.sunlit_nodes if sunlit else settings.eclipse_nodes),
        # pseudospectral only: Radau segments per arc, which must divide num_nodes - 1
        "hp_segments": int(settings.get("sunlit_hp_segments" if sunlit else "eclipse_hp_segments", 1)),
        "fcns": {
            "dynamics": f"{_SOURCE}:dynamics",
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
            # The steering angle MUST be free to wind.  The locally optimal
            # primer rotates roughly once per revolution (Kelly & Bevilacqua
            # Figure 7, where the primer angle sweeps the full circle every
            # orbit), so over an N-revolution transfer theta sweeps ~2*pi*N.
            # Bounding it to +/-2*pi caps the primer at one revolution of
            # winding: beyond that the control saturates, the sail can no longer
            # follow the optimal direction, and the eccentricity is pumped
            # secularly instead of being held at the cap.  The box is kept only
            # to stop the angle running off to infinity.
            "steering_limits": {
                "type": "control_limits",
                "lower": [-2.0 * np.pi * (settings.revolutions + 2)],
                "upper": [2.0 * np.pi * (settings.revolutions + 2)],
            },
            # A sail cannot slew instantaneously.  Without this the minimum-time
            # optimum is bang-bang: the primer flips by ~180 deg between adjacent
            # nodes, which no mesh can represent, the SCP linearisation stays
            # poor and the dynamics defect only falls like 1/N.  The limit is on
            # d(theta)/dL, steering radians per radian of orbital travel.
            "steering_rate_limit": {
                "type": "control_rate_limit",
                "value": [float(settings.max_steering_rate_rad_per_rad)],
            },
        },
        "costs": {},
        "outputs": {},
        "guess": {"type": "propagation"},
    })


def _add_minimum_time_cost(segment: AttrDict, settings: AttrDict) -> None:
    """Attach the minimum-time objective to the final segment.

    Two forms, because they fail in opposite ways:

    ``longitude``
        trajopt's ``min_time``, acting on the independent variable.  It cannot
        be gamed -- the longitude grid carries no virtual-buffer slack -- but it
        is *indifferent* to how far the orbit is over-raised, because raising
        the orbit past the requirement costs no extra longitude.  That leaves a
        flat set of near-optimal solutions for the solver to drift within.

    ``elapsed_time``
        a cost on the elapsed-time state.  Not indifferent: a higher orbit has a
        longer period, so over-raising costs time directly.  Historically this
        form was reducible by pushing elapsed time *backwards* through the
        dynamics' virtual buffer, which produced negative-duration phases; that
        was a symptom of badly scaled states and does not occur once the scaling
        is right, but it is the riskier of the two.

    ``both``
        the ungameable longitude cost, plus the elapsed-time cost at a small
        weight purely to break the tie within the flat set.
    """
    form = str(settings.minimum_time_objective)
    weight = float(settings.elapsed_time_cost_weight)

    if form in ("longitude", "both"):
        segment.costs.minimum_longitude = recursive_attrdict({"type": "min_time"})
    if form in ("elapsed_time", "both"):
        segment.costs.minimum_elapsed_time = recursive_attrdict(
            {"type": "final_state", "idx": 5,
             "w": 1.0 if form == "elapsed_time" else weight}
        )
    if form not in ("longitude", "elapsed_time", "both"):
        raise ValueError(
            f"minimum_time_objective must be 'longitude', 'elapsed_time' or "
            f"'both', got '{form}'"
        )


def build_trajectory_config(settings: AttrDict) -> AttrDict:
    """Expand the compact problem settings into the full multi-segment config."""
    # problem.yaml leaves the mesh to the algorithm on purpose, so that no
    # algorithm silently inherits another one's node counts.
    missing = [key for key in ("sunlit_nodes", "eclipse_nodes") if key not in settings]
    if missing:
        raise ValueError(
            f"problem.{' and problem.'.join(missing)} not set: the mesh belongs to "
            "the algorithm, so set it in the algorithm file (see algorithms/ms.yaml)"
        )
    phases = phase_boundaries(settings)
    x0 = _initial_state(settings)
    initial_longitude = phases[0][0]
    total_longitude = phases[-1][1] - initial_longitude
    constants_geo = float(settings.constants.geo_radius_km)
    window = float(settings.terminator_window_rad)
    freedom = float(settings.final_longitude_freedom_rad)

    # One tolerance for the whole trajectory: the budget is on the accumulated
    # defect, so it has to be shared out over every interval in every phase.
    eps_dynamics = defect_tolerance(settings, phases)

    segments = AttrDict()
    previous_name = None

    for index, (arc_start, arc_stop, sunlit) in enumerate(phases, start=1):
        name = f"phase_{index:03d}_{'sunlit' if sunlit else 'eclipse'}"
        segment = _build_segment(settings, sunlit)
        segment.constraints.dynamics.eps = eps_dynamics
        is_last = index == len(phases)

        # The guess propagates the true dynamics through each arc in turn and
        # hands its endpoint to the next arc, so it is a genuine trajectory:
        # continuous, and with essentially zero dynamics defect.
        progress_start = (arc_start - initial_longitude) / total_longitude
        progress_stop = (arc_stop - initial_longitude) / total_longitude
        segment.guess.update({
            "t_start": float(arc_start),
            "t_stop": float(arc_stop),
            "x_start": x0.tolist() if previous_name is None else "previous",
            "u_start": [_steering_guess(progress_start, settings,
                                        arc_start, initial_longitude)],
            "u_stop": [_steering_guess(progress_stop, settings,
                                       arc_stop, initial_longitude)],
        })

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
            # The sail attitude is physically continuous, and this also pins the
            # steering angle inside eclipse arcs, where it has no effect on the
            # dynamics and would otherwise be a free null direction.
            segment.constraints.control_continuity = recursive_attrdict(
                {"type": "control_continuity", "phase": previous_name}
            )

        if is_last:
            # The trajectory may end anywhere inside the final sunlit arc: this
            # is the freedom the minimum-time cost needs.  The upper bound is
            # capped at the next shadow entry, because past that point the
            # trajectory would fly through an eclipse that has no phase to model
            # it and the sail would keep thrusting in the dark.
            # arc_stop is already the shadow entry (see final_longitude), so it
            # is the hard upper bound: past it the sail would be thrusting in a
            # shadow that has no phase to model it.  The true entry moves later
            # as the orbit is raised -- the half-shadow angle shrinks with 1/r --
            # so the nominal value is the conservative side of the real one.
            end_lower = max(arc_start + _MIN_ARC_RAD, arc_stop - freedom)
            end_upper = arc_stop
        else:
            # An interior arc ends on the shadow terminator.  The equality below
            # places it physically; the window is only a guard that keeps the arc
            # from collapsing or inverting.
            #
            # The equality can be switched off: over this transfer the orbit
            # radius changes by well under 1 %, so the true terminator moves by
            # less than 0.11 deg and the window alone already places the boundary
            # to far better than the discretisation.  Turning it off removes the
            # problem's only nonconvex equality.
            if bool(settings.enforce_shadow_terminator):
                segment.constraints.ends_on_terminator = recursive_attrdict({
                    "type": "final_nonconvex_equality",
                    "fcn": "fcns.shadow_boundary",
                    "scale": [1.0],
                })
            end_lower, end_upper = arc_stop - window, arc_stop + window

        segment.constraints.arc_end_window = recursive_attrdict({
            "type": "final_nonconvex_inequality",
            "fcn": "fcns.phase_end_longitude",
            "lower": np.array([float(end_lower)]),
            "upper": np.array([float(end_upper)]),
            "scale": [2.0 * np.pi],
        })

        if is_last:
            # Graveyard orbit per the IADC guidelines: near-circular, safely
            # above the geostationary belt.  Both are exact second-order cone
            # constraints, so they enter the CVXPY subproblem directly.
            if settings.graveyard.get("buffer_terminal", False):
                # Buffered form for autoscvx-term, which enforces the dynamics
                # without buffers and relaxes the terminal conditions instead.
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
                    {"type": "final_nonconvex_inequality",
                     "fcn": "fcns.terminal_eccentricity",
                     "upper": np.array([params["eccentricity"]]),
                     "scale": [0.02]}
                )
                segment.constraints.final_perigee = recursive_attrdict(
                    {"type": "final_nonconvex_inequality",
                     "fcn": "fcns.terminal_perigee_radius",
                     "lower": np.array([params["perigee"]]),
                     "scale": [1000.0]}
                )
            else:
                segment.constraints.final_eccentricity = recursive_attrdict(
                    {"type": "final_convex_inequality",
                     "fcn": "fcns.terminal_eccentricity_cone"}
                )
                segment.constraints.final_perigee = recursive_attrdict(
                    {"type": "final_convex_inequality",
                     "fcn": "fcns.terminal_perigee_cone"}
                )
            _add_minimum_time_cost(segment, settings)

        segments[name] = segment
        previous_name = name

    return recursive_attrdict({"phases": segments})


class SolarSailProblem:
    """Builds and holds the trajopt method for one value of ``revolutions``.

    Typical use::

        problem = SolarSailProblem(CONFIG_FILE)
        converged = problem.solve()

    To sweep the structural guess without touching the YAML::

        problem = SolarSailProblem(CONFIG_FILE, {"revolutions": 18})

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.
    problem_overrides:
        Optional ``problem.*`` settings applied after loading the YAML.
    method_overrides:
        Optional ``method.*`` settings, deep-merged after loading the YAML, for
        sweeping solver options without editing the file.
    initial_guess:
        Optional ``{phase name: trajopt Guess}``, typically
        :meth:`export_guess` of an earlier solve of the same problem under a
        different discretisation (see :mod:`algorithm`).  Phases left out use
        the propagated guess configured in the problem.
    guess_layouts:
        The :meth:`phase_layouts` of the problem ``initial_guess`` came from.
        When given, every guessed phase is checked against this problem's
        state/control channels before the method is built.
    """

    def __init__(
        self,
        config_path: str | Path,
        problem_overrides: dict | None = None,
        method_overrides: dict | None = None,
        initial_guess: dict | None = None,
        guess_layouts: dict | None = None,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.config = config_loader.load_trajopt_config(str(self.config_path))
        self.settings = deepcopy(self.config.problem)

        # Deep-merged so a nested override such as
        # {"graveyard": {"perigee_altitude_km": 500.0}} keeps its siblings, and
        # routed through recursive_attrdict so nested values stay attribute-
        # accessible the way the rest of this module expects.
        if problem_overrides:
            self.settings = recursive_attrdict(
                tools.deep_merge(dict(self.settings), problem_overrides)
            )

        if method_overrides:
            self.config.method = recursive_attrdict(
                tools.deep_merge(dict(self.config.method), method_overrides)
            )

        self.config.trajectory = build_trajectory_config(self.settings)
        trajectory_class = config_loader.resolve_formulation_trajectory_class(self.config.method)
        self.trajectory = trajectory_class(self.config.trajectory)
        if initial_guess is not None and guess_layouts is not None:
            # trajopt's own flow check: same phase names, channels and units
            source = SimpleNamespace(name="initial_guess", trajectory=initial_guess,
                                     layouts=guess_layouts)
            initial_guess = transfer_guesses(source, self.phase_layouts())
        method_class = config_loader.resolve_scp_method_class(self.config.method)
        self.method = method_class(
            self.config.method, self.trajectory,
            **({} if initial_guess is None else {"initial_guess": initial_guess}),
        )

    @property
    def revolutions(self) -> int:
        return int(self.settings.revolutions)

    @property
    def segments(self):
        """The solved phases (trajopt subproblems), in chain order."""
        return self.method.scp_trajectory.scp_phases

    def solve(self) -> bool:
        """Run SCP until convergence or the iteration budget is exhausted."""
        self.method.solve()
        return bool(self.method._converged)

    def export_guess(self) -> dict:
        """The current iterate as ``{phase name: Guess}``, mesh-independent.

        Feed it to another :class:`SolarSailProblem` as ``initial_guess`` to warm
        start a different discretisation of the same problem: trajopt resamples
        it onto the new mesh (polynomially from collocation, linearly from
        shooting).
        """
        return self.method.export_guess(include_aux=False)

    def phase_layouts(self) -> dict:
        """Per-phase state/control channel layout, for checking a guess transfer."""
        return phase_layouts(self.trajectory, self.config.trajectory)
