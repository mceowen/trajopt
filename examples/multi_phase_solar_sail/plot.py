"""Plot the native solution and its independently replayed trajectory."""

import numpy as np
import matplotlib.pyplot as plt

from trajopt.analysis.plotting import save_figures


def plot_solution(result, replay, settings, directory):
    samples = [t.sample(t.native_tau) for t in result.trajectory.values()]
    nodes = np.concatenate([s.x for s in samples])
    geo, day = settings.constants.geo_radius_km, settings.constants.day_s
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for state, style, label in [(nodes, ".", "Nodes"), (replay["state"], "-", "Chained replay")]:
        eccentricity = np.hypot(state[:, 1], state[:, 2])
        axes[0].plot(
            state[:, 5] / day, (geo + state[:, 0]) / (1 + eccentricity) - geo, style, label=label
        )
        axes[1].plot(state[:, 5] / day, eccentricity, style, label=label)
    axes[0].axhline(
        settings.graveyard.perigee_altitude_km, color="black", ls="--", label="Terminal requirement"
    )
    axes[1].axhline(settings.graveyard.eccentricity_max, color="black", ls="--")
    axes[2].plot(replay["state"][:, 5] / day, replay["control"][:, 0])
    for ax, ylabel in zip(axes, ["Perigee above GEO (km)", "Eccentricity", "Steering angle (rad)"]):
        ax.set(xlabel="Elapsed time (days)", ylabel=ylabel)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    save_figures({"orbit_and_control": figure}, directory, format="png")
    plt.close(figure)
