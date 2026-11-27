# 6-DoF powered descent (Chapters 5 and 6)

Six-degree-of-freedom lander with glide-slope, tilt, angular-rate, gimbal, and thrust limits and a minimum-time objective. `config.yaml` is the shipped case: `mission.yaml` adds the altitude-triggered speed and angular-rate limits of Chapter 5 to `mission/problem.yaml`, solved with AutoSCvx in the standard setting (dynamics buffered) from a straight-line guess on 15 nodes; `python main.py` solves and plots it.

Chapter 5 (outputs go to `plots/<study>/`):

- `run_standalone.py`: the dense AutoSCvx case on its own.
- `run_compare.py` (`compare.yaml`): dense AutoSCvx, Sparse-AutoSCvx (`sparse.yaml`), and its scalar variant on the same problem.
- `run_dispersed.py` (`dispersed.yaml`): the three methods from 20 dispersed guesses plus the nominal one.

Chapter 6, the same problem without the altitude-triggered constraints (`mission_nostc.yaml`):

- `pdg_matrix.py --dispersed 20`: standalone multiple shooting, p-method, and hp-method (`pdg_compare.yaml`) from the straight-line guess and 20 dispersed guesses, with the acceptance test.
- `pdg_handoff.py --seed {p,hp,hp-term} [--seed-budget N]`: a collocation seed stage, converged or stopped after N iterations, handed off to the multiple-shooting final stage.
- `pdg_report.py`: the table and figure of the study.

Flows shipped with the package: `p_to_ms.py`, `p_to_ms_to_hp.py`, and `dt_to_ct.py`, with the stage configurations in `stages/`.
