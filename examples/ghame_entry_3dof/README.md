# GHAME entry (Chapter 6)

Maximum-crossrange entry of the GHAME vehicle (3-DoF, bank angle and angle of attack) with heat-rate, dynamic-pressure, and load constraints. `config.yaml` is the shipped case on 50 nodes with multiple shooting (`mission.yaml`, `method.yaml`); `python main.py` solves and plots it. The problem of the chapter, with all three path constraints, is `mission_all.yaml`, and `compare_all.yaml` lists the transcriptions compared.

Chapter 6 pipeline (outputs go to `plots/`):

- `matrix.py`: tuning screen of the standalone transcriptions (multiple shooting, p-method, hp-method) over mesh and proximal steps, with the replay check of `replay.py` on every converged run. `--dispersed 20` repeats the selected configurations from 20 dispersed guesses; `--ms-nodes`, `--p-nodes`, `--hp-mesh`, and `--hp-knots` override the meshes for the refinement and non-uniform-knot runs; `--config compare_std.yaml` or `compare_term_sl.yaml` runs the straight-line-guess settings (`method_std.yaml`, `method_term_sl.yaml`).
- `choose.py`: selects the reported configuration of each transcription from the screen; `final.py`: reruns the selections three times in fresh processes and keeps the median solve time.
- `handoff.py`: two-stage hybrid methods, a p- or hp-method seed stage (`stages/p_seed.yaml`, `stages/hp_seed.yaml`) resampled as the initial guess of the multiple-shooting final stage (`stages/ms_finish.yaml`). `--seed-budget N` stops the seed stage after N iterations, `--weights` transfers the terminal penalty weights, and `--guess` or `--case` start from a dispersed guess. `handoff_final.py` reruns the selected hybrids three times.
- `margin.py`: the nodal path-limit margin study.
- `report.py`, `dispersed_report.py`, `refine_report.py`, `budget_report.py`: the tables and figures of the chapter from the saved runs.
