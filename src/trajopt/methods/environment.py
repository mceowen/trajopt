"""Record a source hash and environment versions with numerical results."""
from functools import lru_cache
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform


@lru_cache(maxsize=1)
def collect_environment():
    packages = {}
    for name in ('trajopt', 'numpy', 'scipy', 'jax', 'jaxlib', 'cvxpy', 'clarabel'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return dict(python=platform.python_version(), platform=platform.platform(),
                packages=packages, package_source_sha256=digest.hexdigest())
