# Algorithms for the solar-sail problem

The problem is defined once, in [`../problem.yaml`](../problem.yaml): physics,
terminal requirement, initial guess, revolution search. Everything in this
folder is a way of *solving* it. Every script takes any of these files:

```bash
python main.py algorithms/ms.yaml                  # revolution search + figures
python main.py algorithms/p_to_ms.yaml --no-search -n 21
python compare.py algorithms/ms.yaml algorithms/p.yaml algorithms/p_to_ms.yaml --csv study.csv
```

| file | what it is |
| --- | --- |
| `ms.yaml` | RK4 multiple shooting, 12/3 nodes per sunlit/eclipse arc |
| `p.yaml` | flipped Legendre-Radau collocation, one polynomial per arc, 24/4 |
| `hp.yaml` | Radau hp: sunlit arcs split into 4 segments of order 6, 25/4 |
| `p_to_ms.yaml` | flow: p, then shooting warm-started from it |
| `p_to_ms_to_hp.yaml` | flow: p, then shooting, then hp, each warm-started from the one before |
| `method_common.yaml` | method settings every algorithm shares on this problem (buffers, convergence, conic solver) |
| `warm_start.yaml` | method settings for a stage seeded by an earlier one — **use it on every seeded stage** |

`../config.yaml` and `../config-ps.yaml` are aliases of `ms.yaml` and `p.yaml`,
kept so that existing commands and references still work.

## A single-method algorithm

An algorithm file inherits the problem and adds the two things that belong to
the discretisation: a **mesh**, as `problem.*` overrides, and a **method**.

```yaml
inherit: ../problem.yaml

problem:                       # the mesh, and anything else discretisation-dependent
  sunlit_nodes: 12             # required: problem.yaml deliberately has no mesh
  eclipse_nodes: 3
  sunlit_hp_segments: 1        # collocation only; must divide sunlit_nodes - 1
  eclipse_hp_segments: 1
  graveyard:
    perigee_margin_km: 0.3     # nodal-cone margins; size them to the
    eccentricity_margin: 0.0   # discretisation's re-propagation error

method:
  inherit:
    - trajopt/methods/dev/scp_phases/configs/second_order/autosqp.yaml   # SCP flavour
    - trajopt/methods/common/configs/discretize/second_order_p.yaml  # omit for shooting
    - method_common.yaml                                          # always, last
  hyperparams:
    penalties:
      dynamics:
        W: {penalty: True, autotune: True, init: 1.0}               # see below
```

Put the trajopt base configs first and `method_common.yaml` after them; the
file's own keys win over both. Do not inherit one algorithm from another unless
you mean to take *all* of its settings (`hp.yaml` inherits `p.yaml` on purpose).

The settings that have mattered most on this problem, all per algorithm:

| setting | why |
| --- | --- |
| mesh | collocation needs ~2x the nodes of shooting before its *re-propagated* trajectory is accurate; judge it by `max re-propagation error`, not the collocation residual |
| `penalties.dynamics.W.init` | 100 for shooting, 1 for collocation; the wrong value costs two revolutions and looks like a structural floor (FINDINGS.md §8) |
| terminal margins | minimum time parks the nodal state on the cones; the propagated orbit then misses by the discretisation error (FINDINGS.md §10) |

## A flow

A flow solves several algorithms in turn on the same problem, handing each
stage's solution to a later one as its initial guess. The schema is trajopt's
own (`examples/algflow`), plus optional inline overrides:

```yaml
flow:
  stages:
    - name: p                        # unique; used by `from:` and by --set
      config: p.yaml                 # any single-method algorithm file
      problem: {sunlit_nodes: 16}    # optional: this stage only
      method: {hyperparams.convergence.iter_max: 40}   # optional; may `inherit`
      allow_nonconverged: true       # optional: hand on an unconverged iterate

    - name: ms
      config: ms.yaml
      initial_guess:
        from: p                      # an earlier stage
        phases: [...]                # optional subset; the rest get the default guess
      method: {inherit: warm_start.yaml}

problem: {...}                       # optional: every stage
method: {...}                        # optional: every stage
output: {directory: results/my_flow} # optional; default results/<file name>
```

Overrides apply in increasing precedence: the stage's config file, the flow's
`problem`/`method`, the stage's inline `problem`/`method`, then `--set` on the
command line. Dotted keys (`hyperparams.convergence.iter_max: 40`) work anywhere.

The guess is handed over phase by phase as a mesh-independent trajopt `Guess`
and resampled onto the new mesh, so every stage must have the same phases — the
same revolution count — but can have any mesh or discretisation. Only one
stage's compiled problem is alive at a time.

The revolution search, the validation and the figures judge a flow by its last
stage. Earlier stages are validated too (`compare.py` lists them; pass
`--final-only` to skip that and save a re-propagation per stage).

### Warm starts need `warm_start.yaml`

A flow hands over the primal trajectory only: the penalty weights restart at
`W.init` and the duals at zero. With collocation's `W.init` of 1, the first
subproblem of a seeded stage trades dynamics defect for minimum time and walks
away from the guess **to a worse local optimum**. Collocation seeded from a
converged shooting solution at 3 revolutions finished at 2.47 d against 1.98 d
from a cold start, and still validated, so nothing flags it. `warm_start.yaml`
starts the dynamics penalty at 100, and the same stage then converges in 7
iterations to the cold-start optimum. `algorithm.py` prints a warning for any
seeded stage whose dynamics penalty starts below 10.

## Overriding settings without editing files

`--set` works with `main.py` and `compare.py`; values are YAML:

```bash
--set problem.sail.area_to_mass_m2_kg=5            # every stage
--set method.hyperparams.convergence.iter_max=100   # every stage
--set p.problem.sunlit_nodes=16                     # stage `p` only
--set ms.method.hyperparams.discretize.nsub=8       # stage `ms` only
```

A single-method file is a one-stage flow whose stage is named after the file
(`ms`, `p`, `hp`), so `--set p.problem.sunlit_nodes=16` also works on `p.yaml`.

## From Python

```python
from algorithm import Algorithm

algorithm = Algorithm("algorithms/p_to_ms.yaml")
run = algorithm.run(revolutions=21,
                    overrides={"problem": {"sail": {"area_to_mass_m2_kg": 5}},
                               "stages": {"p": {"problem": {"sunlit_nodes": 16}}}})
run.print_summary()
for stage in run.stages:                 # status, iterations, timings, solution, report
    print(stage.name, stage.status, stage.timings, stage.report["numerically_feasible"])
```

`Run.feasible` is true when every stage ran and the last one validates.
`algorithm.run(..., keep_problem=True)` leaves the final stage's
`SolarSailProblem` on `run.problem` for inspection. Drop it when you're done,
since it holds the compiled problem.

## Costs to plan around

Measured at 21 revolutions on a 64 GB machine (FINDINGS.md §10):

* Every stage pays its own setup: shooting spends ~5.5 min compiling JAX kernels
  (once per phase), collocation ~1 min. A flow stage costs this again even when
  it then converges in a few iterations.
* Collocation's one-off CVXPY canonicalisation peaks at ~33 GB at 24/4 nodes.
  Stages run one at a time, but each must fit on its own.
* A revolution count that cannot be reached runs to `iter_max` (250).
