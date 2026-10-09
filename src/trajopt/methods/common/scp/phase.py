from trajopt.methods.common.scp.subproblem import Subproblem


class SCPPhase(Subproblem):
    """One phase's SCP subproblem, extended with a boundary-condition check that only
    applies when several phases are chained together (see build_cross_phase on
    the continuity constraint types for the other half of that link)."""

    def _validate_time_config(self, initial, final) -> None:
        if initial is not None and self.find_constraint("time_continuity") is not None:
            raise ValueError(
                f"phase '{self.name}' declares both initial_time and time_continuity; "
                "its start epoch comes from the preceding phase, so drop the initial_time"
            )

    def inherits_start_epoch(self) -> bool:
        """True when a preceding phase sets this phase's start time."""
        return any(self.find_constraint(t) is not None
                   for t in ("time_continuity", "full_continuity"))

    def _anchor_start_time(self) -> bool:
        # the ps mesh is built around a fixed node 0
        return self.hyperparams.discretize.mode == "ps" or not self.inherits_start_epoch()
