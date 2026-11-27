# Quadrotor (Chapter 1)

Planar quadrotor with a no-fly zone, the first benchmark of the AutoSCvx chapter. `config.yaml` is the shipped case: Algorithm Variant 3b in the standard setting (dynamics buffered) from the waypoint guess on 20 nodes; `python main.py` solves and plots it.

Chapter drivers (outputs go to `plots/<study>/`):

- `run_compare.py <yaml>`: nominal comparison of the algorithm variants. `compare.yaml` is the standard setting, `compare_term.yaml` the terminal-buffered setting, `compare_std_prop.yaml` and `compare_term_wp.yaml` the two remaining buffering/guess cells.
- `run_dispersed.py <yaml>`: the same variants from 20 dispersed guesses plus the nominal one (`dispersed.yaml`, `dispersed_term.yaml`, `dispersed_std_prop.yaml`, `dispersed_term_wp.yaml`, `dispersed_tuned.yaml`).
- `run_sweep.py` (`sweep_std.yaml`, `sweep_term.yaml`): dual-step sweep of Variant 1; `screen_v1.py`: Variant 1 over dual and primal steps.
- `plot_nominal.py`, `plot_convergence.py`, `plot_weights.py`, `plot_dispersed.py`, `plot_sweep.py`: the chapter figures.
- `make_tables.py <out.tex>`: the chapter's result tables from the saved runs of this example and of `../unicycle`.

The `variants*.yaml` files define the five configurations (Variants 1, 2a, 2b, 3a, 3b) that the `compare*` and `dispersed*` files select, and `method*.yaml` hold the method settings of each setting.
