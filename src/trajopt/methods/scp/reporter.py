"""Console output for the SCP solve loop."""

_SEP = " │ "


class _Column:
    """One table column: its key, header text, width and number format."""

    def __init__(self, key, header, width, kind):
        self.key = key
        self.header = header
        self.width = width
        self.kind = kind

    def format_value(self, value):
        if value is None:
            text = "—"
        elif self.kind == "int":
            text = f"{int(value):d}"
        elif self.kind == "f2":
            text = f"{value:.2f}"
        elif self.kind == "f3":
            text = f"{value:.3f}"
        elif self.kind == "f4":
            text = f"{value:.4f}"
        elif self.kind == "sci":
            text = f"{value:.4e}"
        elif self.kind == "g":
            text = f"{value:.8g}"
        else:  # "str"
            text = str(value)
        return _center(text, self.width)

    def format_header(self):
        return _center(self.header, self.width)


def _center(text, width):
    """Center text in width, truncating it if it does not fit."""
    if len(text) > width:
        text = text[:width]
    return f"{text:^{width}}"


# The residual columns hold the ratio metric/eps, which reaches 1 at convergence.
_BASE_COLUMNS = [
    _Column("iter",    "Iter",        5,  "int"),
    _Column("disc",    "Disc [ms]",   10, "f2"),
    _Column("solve",   "Solve [ms]",  10, "f2"),
    _Column("parse",   "Parse [ms]",  10, "f2"),
    _Column("dx",      "dx/eps",      12, "sci"),
    _Column("vb_ineq", "vb_ineq/eps", 12, "sci"),
    _Column("vb_eq",   "vb_eq/eps",   12, "sci"),
    _Column("vb_term", "vb_term/eps", 12, "sci"),
    _Column("vb_dyn",  "vb_dyn/eps",  12, "sci"),
    _Column("status",  "Status",      12, "str"),
    _Column("alpha",   "alpha",       8,  "f4"),
    _Column("tof",     "ToF [s]",     10, "f3"),
    _Column("cost",    "Cost",        14, "g"),
    _Column("penalty", "Penalty",     14, "g"),
]

_SEG_COLUMN = _Column("seg", "Segment", 12, "str")


class SolveReporter:
    """Prints the SCP solve progress table. multi adds a Segment column, quiet silences it."""

    def __init__(self, *, multi=False, quiet=False):
        self.quiet = quiet
        self.columns = ([_SEG_COLUMN] + _BASE_COLUMNS) if multi else list(_BASE_COLUMNS)

    def _header_line(self):
        return _SEP.join(col.format_header() for col in self.columns)

    def message(self, text=""):
        """Print a single line of text."""
        if not self.quiet:
            print(text)

    def subproblem_stats(self, *, num_segments, num_params, num_constraints, is_dpp):
        if self.quiet:
            return
        print("subproblem stats:")
        print("------------------------------------------------------------")
        print(f"total number of segments: {num_segments}")
        print(f"total number of cvxpy parameters: {num_params}")
        print(f"total number of cvxpy constraints: {num_constraints}")
        print(f"is DPP: {is_dpp}")

    def header(self):
        if self.quiet:
            return
        line = self._header_line()
        divider = "─" * len(line)
        print(divider)
        print(line)
        print(divider)

    def row(self, iter_data, segment_name=None):
        if self.quiet:
            return
        values = self._extract(iter_data, segment_name)
        cells = [col.format_value(values.get(col.key)) for col in self.columns]
        print(_SEP.join(cells))

    def footer(self, *, reason=None, total_ms=0.0, disc_ms=0.0, solve_ms=0.0):
        if self.quiet:
            return
        print("─" * len(self._header_line()))
        if reason:
            print(reason)
        print(f"Total SCP time: {total_ms:.1f} ms (discretize: {disc_ms:.1f}, solve: {solve_ms:.1f})")

    @staticmethod
    def _extract(iter_data, segment_name):
        chk = iter_data.chk
        return {
            "seg":     segment_name,
            "iter":    int(getattr(iter_data, "iter_num")),
            "disc":    float(getattr(iter_data, "discretization_time")),
            "solve":   float(getattr(iter_data, "solve_time")),
            "parse":   float(getattr(iter_data, "parse_time")),
            "dx":      float(getattr(chk, "dz")),
            "vb_ineq": float(getattr(chk, "nonconvex_inequality")),
            "vb_eq":   float(getattr(chk, "nonconvex_equality")),
            "vb_term": float(getattr(chk, "final_state")),
            "vb_dyn":  float(getattr(chk, "dynamics")),
            "status":  str(getattr(iter_data, "status")),
            "alpha":   float(getattr(iter_data, "alpha", 1.0)),
            "tof":     float(getattr(iter_data, "T_opt")),
            "cost":    float(getattr(iter_data, "cost")),
            "penalty": float(getattr(iter_data, "penalty_cost", 0.0)),
        }
