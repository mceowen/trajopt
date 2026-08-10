"""What analyze() returns, in SI units. Every container also takes dict access.

    MissionResult
      .runs_by_method: {method: [RunResult, ...]}   run 0 is the nominal case
        RunResult
          .iter_data_list: [Iterate]
          .solver_iters                             {segment: per-iteration solver data}
          .final -> Iterate
            Iterate
              t_opt / x_opt / u_opt                 values at the nodes
              t_nl  / x_nl  / u_nl                  propagated trajectory
              t_init_nl / x_init_nl / u_init_nl     initial guess
              outputs                               {name: opt, nl_prop, init_guess, limits, quivers, meta}
              channels                              {name: output.opt}
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import cached_property
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


def _as_column(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 2 and values.shape[1] == 1:
        return values[:, 0]
    return values


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
    outputs: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Iterate":
        """Build an Iterate from a mapping of the same field names."""
        return cls(**{f.name: data[f.name] for f in fields(cls)})

    @cached_property
    def channels(self) -> AttrDict:
        """Every output's node values keyed by name, single-column ones as 1-D."""
        return AttrDict({
            name: _as_column(output.opt) for name, output in self.outputs.items()
        })


@dataclass(frozen=True)
class RunResult(_MappingShim):
    """One solve: its propagated iterates plus the per-segment solver data."""

    iter_data_list: list[Iterate]
    solver_iters: Mapping[str, list]

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
