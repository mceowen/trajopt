# Installation

`trajopt` requires **Python 3.11 or newer**.

## From source

Clone the repository and install the package:

```bash
git clone https://github.com/skye/trajopt.git
cd trajopt
pip install .
```

This installs the core runtime dependencies: `numpy`, `scipy`, `cvxpy`, `jax`,
`sympy`, `diffrax`, `matplotlib`, `pyyaml`, and the conic solvers `qoco` and
`ecos`.

## Development install

To work on `trajopt` itself, install the editable package with the development
tools (`pytest`, `black`, `mypy`, `ruff`):

```bash
pip install -e .[dev]
```

## Building the documentation

To build this documentation site locally:

```bash
pip install .[docs]
sphinx-build -b html docs docs/_build/html
```

The rendered site is written to `docs/_build/html/index.html`.
