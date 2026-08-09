"""Low-level Matplotlib helpers for plotting SCP iteration data."""

from typing import Any
import numpy as np

import matplotlib.pyplot as plt
import trajopt.utils.tools as tools
plt.rcParams['text.usetex'] = False

class SCVXPLOTS:
    """Plotting helper for SCP trajectory data organized by method, run, and iteration."""

    def __init__(self, data: dict) -> None:
        """Initialize with the nested results dict (method → runs → iters)."""
        self.data = data

        self.base_pen = {"frgba": [0, 0, 0, 0.1], "lrgba": [0, 0, 0, 0.1], "lw": 2, "ls": "-", "msty": "", "msz": 1}

        self.legends: dict = {}

    ########### BASIC 2D-PLOTTING ###############
    def addPlot2D(self, ax: Any, pen: dict = {}, ins: dict = {}) -> None:
        """Plot 2-D (x vs y) data from a selected method/run/iter slice onto ax."""
        # extract mandatory (method, run, and iters) from ins
        method_name = ins["method"]
        run = ins["run"]
        iters = ins["iters"]

        # x and y data
        x_path = ins["x"]
        x_idx = ins.get("x_idx", slice(None))

        y_path = ins["y"]
        y_idx = ins.get("y_idx", slice(None))

        # legend and label
        leg = ins["legend"]
        label = ins["label"]

        # pen data
        pen = {**self.base_pen, **pen}
        frgba = pen["frgba"]
        lrgba = pen["lrgba"]
        lw = pen["lw"]
        ls = pen["ls"]
        msty = pen["msty"]
        msz = pen["msz"]

        iter_data_list = np.array(self.data[method_name]["runs"][run]["iters"])
        selected_iters = iter_data_list[iters]
        n_iters = len(selected_iters)

        for i, iter_data in enumerate(selected_iters):
            # get x and y data from iter_data
            x_data = tools.get_from_path(iter_data, x_path)
            y_data = tools.get_from_path(iter_data, y_path)

            if x_data.size == 0 or y_data.size == 0:
                continue

            # force everything to be 2D for plotting, dont change the time axis
            if x_data.ndim == 1:
                x_data = x_data[:, np.newaxis]
            if y_data.ndim == 1:
                y_data = y_data[:, np.newaxis]

            # hack to get y_data with N-1 points to match x_data
            if y_data.shape[0] == x_data.shape[0] - 1:
                x_data = x_data[:-1]

            first_frac = pen.get("first_frac", 0.2)
            if n_iters > 1:
                t = i / (n_iters - 1)
                alpha = lrgba[3] * (first_frac + (1.0 - first_frac) * t)
            else:
                alpha = lrgba[3]

            ax.plot(
                x_data[:, x_idx],
                y_data[:, y_idx],
                color=lrgba[:3],
                alpha=alpha,
                linewidth=lw,
                linestyle=ls,
                marker=msty,
                markersize=msz,
            )

    ########### BASIC 3D-PLOTTING ###############
    def addPlot3D(self, ax: Any, pen: dict = {}, ins: dict = {}) -> None:
        """Plot 3-D (x, y, z) data from a selected method/run/iter slice onto a 3-D axes."""
        method_name = ins.get("method", next(iter(self.data.keys())))
        iters = ins.get("iters", slice(None, -1))
        run = ins.get("run", 0)

        # x, y, and z data
        x_path = ins["x"]
        x_idx = ins.get("x_idx", slice(None))
        y_path = ins["y"]
        y_idx = ins.get("y_idx", slice(None))
        z_path = ins["z"]
        z_idx = ins.get("z_idx", slice(None))

        # legend and label
        leg = ins["legend"]
        label = ins["label"]

        # pen data
        pen = {**self.base_pen, **pen}
        frgba = pen["frgba"]
        lrgba = pen["lrgba"]
        lw = pen["lw"]
        ls = pen["ls"]
        msty = pen["msty"]
        msz = pen["msz"]

        iter_data_list = np.array(self.data[method_name]["runs"][run]["iters"])

        for i, iter_data in enumerate(iter_data_list[iters]):
            # get x and y data from iter_data
            x_data = tools.get_from_path(iter_data, x_path)
            y_data = tools.get_from_path(iter_data, y_path)
            z_data = tools.get_from_path(iter_data, z_path)

            # force everything to be 2D for plotting, dont change the time axis
            if x_data.ndim == 1:
                x_data = x_data[:, np.newaxis]
            if y_data.ndim == 1:
                y_data = y_data[:, np.newaxis]
            if z_data.ndim == 1:
                z_data = z_data[:, np.newaxis]

            ax.plot(
                x_data[:, x_idx],
                y_data[:, y_idx],
                z_data[:, z_idx],
                color=lrgba[:3],
                alpha=lrgba[3],
                linewidth=lw,
                linestyle=ls,
                marker=msty,
                markersize=msz,
            )

    ########### BASIC 2D-PLOTTING over iterations ###############
    def addPlot2D_iters(self, ax: Any, pen: dict = {}, ins: dict = {}) -> None:
        """Plot a scalar quantity across SCP iterations (iteration number on x-axis)."""
        method_name = ins.get("method", next(iter(self.data.keys())))
        run = ins.get("run", 0)

        y_path = ins["y"]
        y_idx = ins.get("y_idx", slice(None))

        # legend and label
        leg = ins["legend"]
        label = ins["label"]

        # pen data
        pen = {**self.base_pen, **pen}
        frgba = pen["frgba"]
        lrgba = pen["lrgba"]
        lw = pen["lw"]
        ls = pen["ls"]
        msty = pen["msty"]
        msz = pen["msz"]

        iter_data_list = np.array(self.data[method_name]["runs"][run]["iters"][1:])
        last_iter_data = iter_data_list[-1]

        # get x and y data from last iter_data
        y_data_last = tools.get_from_path(last_iter_data, y_path)

        if y_data_last.size == 0:
            return

        if y_data_last.ndim == 1:
            y_data_last = y_data_last[np.newaxis, :]

        y_data = np.zeros((len(iter_data_list), y_data_last.shape[1]))

        for i, iter_data in enumerate(iter_data_list):
            iter_y_data = tools.get_from_path(iter_data, y_path)
            if iter_y_data.size == 0:
                continue
            if iter_y_data.ndim == 1:
                y_data[i, :] = iter_y_data

            elif iter_y_data.ndim == 2:
                y_data[i, :] = np.max(iter_y_data, axis=0)

        n = len(iter_data_list)
        xs = np.arange(1, n + 1)
        ax.plot(xs, y_data[:, y_idx], color=lrgba[:3], alpha=lrgba[3], linewidth=lw, linestyle=ls, marker=msty, markersize=msz)

    ######## LABELS AND LEGENDS ############
    def setTicks(self, ax: Any, x: bool = False, y: bool = False, ins: dict = {}) -> None:
        """Apply tick_params to x and/or y axis of ax."""
        if x:
            ax.tick_params(**ins)
        if y:
            ax.tick_params(**ins)

    def setLabels(self, ax: Any, xlabel: str = "", ylabel: str = "", ins: dict = {}) -> None:
        """Set x and y axis labels on ax."""
        ax.set_xlabel(xlabel, **ins)
        ax.set_ylabel(ylabel, **ins)

    def setTitle(self, ax: Any, title: str = "", ins: dict = {}) -> None:
        """Set the title on ax."""
        ax.set_title(title, **ins)

    def addLegend(self, ax: Any, leg: str, labels: list = [], ins: dict = {}) -> None:
        """Add a legend to ax from the stored legend group leg."""
        if len(labels) == 0:
            labels = list(self.legends[leg])
        handles = [self.legends[leg][label] for label in labels]
        ax.legend(handles, labels, **ins)

    def dumpLegend(self, leg: str) -> None:
        """Clear all entries from legend group leg."""
        self.legends[leg] = {}

    ########## CONSTRUCT SUBPLOTS ##########
    def genGridTags(self, fig: Any, typ: Any = None, params: dict = {}) -> dict:
        """Generate axes from a spec grid using createGrid."""
        return self.createGrid(fig, typ=typ, grid=self.specGrid(typ=typ, params=params))

    def createGrid(self, fig: Any, typ: str = "manual", grid: dict = {}, ins: dict = {}) -> dict:
        """Create a flat dict of 2-D axes from a manually specified grid layout."""
        plt_typ = "2D"
        if "plt_typ" in ins:
            plttyp = ins["plt_typ"]
        if typ == "manual":
            axs = {}
            for tag in grid:
                if plt_typ == "3d":
                    axs[tag] = fig.add_axes(grid[tag], projection="3d")
                else:
                    axs[tag] = fig.add_axes(grid[tag])
        return axs

    def createGrid2(self, fig: Any, typ: str = "manual", grid: dict = {}, ins: dict = {}) -> dict:
        """Create a dict of axes supporting mixed 2-D/3-D subplots from a manual grid layout."""
        plttyps = {}
        if "plt_typs" in ins:
            plttyps = ins["plt_typs"]

        pad_3d = ins.get("pad_3d", 0.08)

        if typ == "manual":
            axs = {}
            for tag in grid:
                if tag in plttyps and plttyps[tag] == "3D":
                    x, y, w, h = grid[tag]
                    axs[tag] = fig.add_axes([x - pad_3d, y, w + pad_3d, h], projection="3d")
                else:
                    axs[tag] = fig.add_axes(grid[tag])
        return axs
