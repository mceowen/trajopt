"""Guess export and reset for SCP methods."""
from trajopt.methods.common.initial_guess import (
    dimensional_samples, export_guess, phase_guesses,
)


class GuessLifecycle:
    def _guess_subproblems(self):
        trajectory = self.scp_trajectory
        return (trajectory.scp_phases if hasattr(trajectory, 'scp_phases')
                else trajectory.scp_subproblems)

    def export_guess(self):
        """Snapshot each phase by name."""
        return {name: export_guess(p) for name, p in self._guess_subproblems().items()}

    def reset(self, *, initial_guess=None):
        """Validate all guesses, then reset iteration state while retaining compiled objects."""
        phases = self._guess_subproblems()
        supplied = phase_guesses(initial_guess, phases)
        prepared = {}
        previous = None
        for name, phase in phases.items():
            guess = phase.prepare_guess(supplied.get(name), previous)
            prepared[name] = guess
            previous = dimensional_samples(phase.phase, guess.z, guess.nu)
        for name, phase in phases.items():
            phase.install_initial_guess(prepared[name])
        for variable in self.cp_subproblem.variables():
            variable.value = None
        self._converged = False
