# Solar-sail transfer from GEO

Run from the repository root:

```sh
.venv/bin/python examples/multi_phase_solar_sail/main.py --output plots/solar_sail --plot
```

The default uses first-order AutoSCvx with multiple shooting. It starts from a
propagated steering guess. The output directory contains a native `SolveResult`
(`solution.json` and `solution.npz`), `validation.json`, and the optional plot.
Load the solution with `trajopt.common.persistence.load_result`.

For the pure-p comparison, pass `examples/multi_phase_solar_sail/p.yaml` as the
positional configuration argument. This is an experimental baseline; convergence
is not assumed. The command exits with status 1 when the validation checks fail.

The model and phase builder were ported from `origin/riccardo-branch` at
`85ab54a8a2726bdac8154bdce3af8ea4338734ec`. They use native problem inputs,
initial-guess propagation, SCP, persistence, and trajectory replay. The example
rejects second-order solver settings.

The state is `[p - r_GEO, f, g, h, k, elapsed_time]`, with distances in km and
elapsed time in seconds. The independent variable is true longitude in radians.
The model includes Earth point-mass gravity, solar radiation pressure, a uniformly
rotating Sun direction, and cylindrical eclipses. The default has 21 nominal
revolutions and 41 alternating sunlit/eclipse phases. `--revolutions N` changes
that fixed phase structure; the example does not search over revolution counts.

The objective minimizes final elapsed time. The required terminal perigee is
250 km above GEO, with eccentricity at most 0.003. The default adds small margins
to the optimization constraints. Validation replays the controls from the exact
initial orbit through the complete phase chain without resetting the state at
phase boundaries. It checks the terminal requirements without tolerance slack,
the accumulated semilatus-rectum defect budget, solver convergence, phase
continuity, and positive phase durations. These checks do not certify
continuous-time path feasibility.

The selected default reproduces our retained current-library MS solution. The
comparison with Skye's committed first-order settings, candidate configurations,
and reproduction commands are retained in
`local/scitech2027_pseudospectral_scp/solar_sail/port_comparison.json`.
