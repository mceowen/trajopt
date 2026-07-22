"""Typed result of :meth:`TrajectoryAnalyzer.analyze`::

    MissionResult                    # what analyze() returns
      .runs_by_method: {method: [RunResult, ...]}
        RunResult                    # one solve; run 0 is the nominal case
          .iter_data_list: [Iterate]
          .scp_iters                 # raw solver-side iterate data
          .final -> Iterate          # last iterate
            Iterate                  # one SCP iterate, in SI units
              t_opt / x_opt / u_opt              values at the nodes
              t_nl  / x_nl  / u_nl               propagated trajectory
              t_init_nl / x_init_nl / u_init_nl  initial guess
              trajplot_data                      group -> name -> payload

All arrays are in SI units. ``x_*`` holds the physical states only, with time
in ``t_*``. Every container also supports dict-style access, so
``iterate["x_opt"]`` works alongside ``iterate.x_opt``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping

import numpy as np

from trajopt.utils.tools import AttrDict


class _MappingShim:
    """Adds read-only dict-style access to a dataclass's fields."""

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self):
        return [f.name for f in fields(self)]

    def __contains__(self, key: str) -> bool:
        return any(f.name == key for f in fields(self))

    def __iter__(self):
        return iter(self.keys())


@dataclass(frozen=True)
class Iterate(_MappingShim):
    """One SCP iterate in SI units."""

    iter_num: int
    t_opt: np.ndarray
    x_opt: np.ndarray
    u_opt: np.ndarray
    t_nl: np.ndarray
    x_nl: np.ndarray
    u_nl: np.ndarray
    t_init_nl: np.ndarray
    x_init_nl: np.ndarray
    u_init_nl: np.ndarray
    trajplot_data: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Iterate":
        """Build an Iterate from a mapping of the same field names."""
        return cls(**{f.name: data[f.name] for f in fields(cls)})


@dataclass(frozen=True)
class RunResult(_MappingShim):
    """One solve: its propagated iterates plus raw solver data."""

    iter_data_list: list[Iterate]
    scp_iters: Any

    @property
    def final(self) -> Iterate:
        """Return the last iterate."""
        return self.iter_data_list[-1]


@dataclass(frozen=True)
class MissionResult:
    """Maps each method name to its runs. Run 0 is the nominal case."""

    runs_by_method: dict[str, list[RunResult]]

    def __getitem__(self, method: str) -> AttrDict:
        return AttrDict({"runs": self.runs_by_method[method]})

    def get(self, method: str, default: Any = None) -> Any:
        if method in self.runs_by_method:
            return self[method]
        return default

    def keys(self):
        return self.runs_by_method.keys()

    def values(self):
        return [self[m] for m in self.runs_by_method]

    def items(self):
        return [(m, self[m]) for m in self.runs_by_method]

    def __iter__(self):
        return iter(self.runs_by_method)

    def __contains__(self, method: str) -> bool:
        return method in self.runs_by_method

    def __len__(self) -> int:
        return len(self.runs_by_method)

    def primary_run(self, method: str | None = None) -> RunResult:
        """Return run 0 of method, or of the first method."""
        if method is not None and method in self.runs_by_method:
            name = method
        else:
            name = next(iter(self.runs_by_method))
        return self.runs_by_method[name][0]

    def final_iterate(self, method: str | None = None) -> Iterate:
        """Return the final iterate of the primary run."""
        return self.primary_run(method).final
