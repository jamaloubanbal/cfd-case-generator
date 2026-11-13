"""
Validation benchmarks as tests: generated cases must reproduce analytical and
published reference results within tolerance.

Slow (≈ 12 min in total) and needs OpenFOAM:

    source /usr/share/openfoam/etc/bashrc
    pytest -m benchmark -v

The same benchmarks produce the report in docs/validation/ via ``python -m validation.run``.
"""

import os
import shutil

import pytest

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.skipif(
        not (os.environ.get("WM_PROJECT_DIR") and shutil.which("blockMesh")),
        reason="OpenFOAM environment not sourced",
    ),
]

pytest.importorskip("matplotlib")
from validation.benchmarks import BENCHMARKS  # noqa: E402
from validation.run import run_benchmark  # noqa: E402


@pytest.mark.parametrize("key", list(BENCHMARKS))
def test_benchmark_within_tolerance(key, tmp_path):
    result = run_benchmark(BENCHMARKS[key], tmp_path)
    failures = [
        f"{m.name}: {m.value:.4g} vs {m.reference:.4g} ({m.error * 100:+.2f} %, tol ±{m.tolerance * 100:g} %)"
        for m in result["metrics"] if not m.passed
    ]
    assert not failures, "\n".join(failures)
