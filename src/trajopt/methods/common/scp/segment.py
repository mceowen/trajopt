from trajopt.methods.common.scp.subproblem import SCPSubproblem


class SCPSegment(SCPSubproblem):
    """Subproblem for one phase of a multi-phase trajectory."""

    def _validate_time_config(self, initial, final) -> None:
        if initial is not None and self.find_constraint("time_continuity") is not None:
            raise ValueError(
                f"segment '{self.name}' declares both initial_time and time_continuity; "
                "its start epoch comes from the preceding segment, so drop the initial_time"
            )
