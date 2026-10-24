"""Physical problem definitions with immutable copies of the resolved inputs.

These definitions accept the existing mission and model configurations.
:mod:`trajopt.methods.prepare` constructs the numerical problem.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from trajopt.utils._snapshot import freeze, thaw
from trajopt.utils.tools import AttrDict, recursive_attrdict


CONTINUITY_TYPES = frozenset({
    "state_continuity", "control_continuity", "time_continuity", "full_continuity",
})


@dataclass(frozen=True)
class SegmentSpec:
    """A named physical segment with an immutable copy of its resolved inputs."""

    name: str
    inputs: Mapping
    agent: str | None = None

    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError("A segment requires a nonempty string name")
        object.__setattr__(self, "inputs", freeze(self.inputs))

    @property
    def config(self):
        """Return a mutable configuration copy for physical model construction."""
        return recursive_attrdict(thaw(self.inputs))

    def build(self):
        from trajopt.formulations.common.problem import Problem
        return Problem(self.config, name=self.name)


@dataclass(frozen=True)
class CouplingSpec:
    """A physical relationship between two named segments.

    Continuity uses the target segment's channel indices by default. For
    different channel orders, set ``source_idx`` and ``target_idx``. For full
    continuity, set ``state_map`` and ``control_map`` instead.

    ``scale`` sets the residual scale in physical units. If omitted, the
    target segment's scales determine the normalized tolerances.
    """

    name: str
    kind: str
    source: str
    target: str
    inputs: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in CONTINUITY_TYPES:
            raise ValueError(f"Unsupported coupling kind {self.kind!r}")
        if self.source == self.target:
            raise ValueError(f"Coupling {self.name!r} must connect distinct segments")
        object.__setattr__(self, "inputs", freeze(self.inputs))

    @property
    def config(self):
        config = recursive_attrdict(thaw(self.inputs))
        config.update(name=self.name, type=self.kind, phase=self.source)
        return config


@dataclass(frozen=True)
class ProblemSpec:
    """Physical segments and couplings with an immutable copy of their inputs."""

    segments: Mapping[str, SegmentSpec]
    couplings: tuple[CouplingSpec, ...] = ()
    metadata: Mapping = field(default_factory=dict)
    flat: bool = False

    def __post_init__(self):
        segments = dict(self.segments)
        if not segments:
            raise ValueError("A problem requires at least one segment")
        for name, segment in segments.items():
            if name != segment.name:
                raise ValueError(f"Segment key {name!r} differs from its name {segment.name!r}")
        couplings = tuple(self.couplings)
        seen = set()
        for coupling in couplings:
            if coupling.source not in segments or coupling.target not in segments:
                raise ValueError(f"Coupling {coupling.name!r} references an unknown segment")
            key = (coupling.target, coupling.name)
            if key in seen:
                raise ValueError(f"Duplicate coupling {coupling.target}/{coupling.name}")
            seen.add(key)
        object.__setattr__(self, "segments", MappingProxyType(segments))
        object.__setattr__(self, "couplings", couplings)
        object.__setattr__(self, "metadata", freeze(self.metadata))
        self.validate_time_constraints()

    @classmethod
    def from_config(cls, config):
        """Build a specification from a resolved flat or multiphase configuration."""
        if isinstance(config, cls):
            return config
        if not isinstance(config, Mapping):
            raise TypeError("A physical problem must be a ProblemSpec or configuration mapping")
        config = config.get("trajectory", config)
        flat = "phases" not in config
        configured = {config.get("name", "main"): config} if flat else config["phases"]
        segments, couplings = {}, []
        for name, source in configured.items():
            cfg = recursive_attrdict(thaw(freeze(source)))
            constraints = cfg.setdefault("constraints", AttrDict())
            for key, constraint in list(constraints.items()):
                kind = constraint.get("type")
                if kind in CONTINUITY_TYPES:
                    if "phase" not in constraint:
                        raise ValueError(f"Coupling {name}/{key} requires a source phase")
                    couplings.append(CouplingSpec(key, kind, constraint["phase"], name, constraint))
                    del constraints[key]
            segments[name] = SegmentSpec(name, cfg, agent=source.get("agent"))
        metadata = {} if flat else {key: value for key, value in config.items() if key != "phases"}
        return cls(segments, tuple(couplings), metadata, flat=flat)

    def validate_time_constraints(self):
        time_links = [c for c in self.couplings
                      if c.kind in {"time_continuity", "full_continuity"}]
        inherited = {c.target for c in time_links}
        for name in inherited:
            constraints = self.segments[name].inputs.get("constraints", {})
            if any(c.get("type") == "initial_time" for c in constraints.values()):
                raise ValueError(
                    f"segment '{name}' declares both initial_time and time continuity; "
                    "its start epoch comes from the preceding segment, so drop the initial_time"
                )
        # A state-only cycle can be a periodic boundary condition. A cycle of
        # inherited epochs cannot contain positive-duration trajectory segments.
        graph = {name: [] for name in self.segments}
        for coupling in time_links:
            graph[coupling.source].append(coupling.target)
        visiting, finished = set(), set()

        def visit(name):
            if name in visiting:
                raise ValueError(f"Cyclic time continuity at segment {name!r} is incompatible with positive durations")
            if name in finished:
                return
            visiting.add(name)
            for successor in graph[name]:
                visit(successor)
            visiting.remove(name)
            finished.add(name)

        for name in graph:
            visit(name)

    @property
    def config(self):
        """Return a mutable configuration copy, including the segment couplings."""
        phases = AttrDict({name: segment.config for name, segment in self.segments.items()})
        for coupling in self.couplings:
            phases[coupling.target].setdefault("constraints", AttrDict())[coupling.name] = coupling.config
        if self.flat and len(phases) == 1:
            return next(iter(phases.values()))
        config = recursive_attrdict(thaw(self.metadata))
        config.phases = phases
        return config

    def build_segments(self):
        """Build one physical model per segment, including its declared couplings."""
        from trajopt.formulations.common.problem import Problem
        configs = {name: segment.config for name, segment in self.segments.items()}
        for coupling in self.couplings:
            configs[coupling.target].setdefault('constraints', AttrDict())[coupling.name] = coupling.config
        return AttrDict({name: Problem(config, name=name) for name, config in configs.items()})


def normalize(config):
    """Split a resolved run configuration into problem and method specifications."""
    from trajopt.methods.specs import MethodSpec
    return ProblemSpec.from_config(config), MethodSpec.from_config(config.get("method", config))
