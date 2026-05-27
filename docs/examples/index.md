# Examples

Each example is a self-contained Jupyter notebook that sets up a mission,
model, and method, then solves the trajectory optimization problem. The
notebooks are rendered here without execution; run them locally to reproduce
the outputs.

```{toctree}
:maxdepth: 1

notebooks/cartpole/main
notebooks/lander_6dof/main
notebooks/msl_entry_3dof/main
notebooks/msrl_entry_3dof_entry/main
notebooks/perching_glider_tedrake/main
notebooks/quadrotor_3dof/main
notebooks/quadrotor_3dof/main_ps
notebooks/rlv_entry_3dof/main
notebooks/starship_flip_malyuta/main
notebooks/vtol1_entry_3dof/main
```

## Example summaries

| Example | Description |
|---|---|
| `cartpole` | Classic cart-pole swing-up. |
| `lander_6dof` | Powered-descent landing with 6-DOF rigid-body dynamics. |
| `msl_entry_3dof` | Mars Science Laboratory atmospheric entry (3-DOF). |
| `msrl_entry_3dof_entry` | Mars Sample Return Lander atmospheric entry (3-DOF). |
| `perching_glider_tedrake` | Perching glider maneuver (Tedrake benchmark). |
| `quadrotor_3dof` | Quadrotor trajectory optimization. `main_ps` uses the pseudospectral method. |
| `rlv_entry_3dof` | Reusable launch vehicle entry (3-DOF). |
| `starship_flip_malyuta` | Starship flip-and-land maneuver (Malyuta benchmark). |
| `vtol1_entry_3dof` | VTOL vehicle entry (3-DOF). |
