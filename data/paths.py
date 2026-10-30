"""Locate the benchmark checkout next to this project."""

from pathlib import Path


def _find_benchmark_root():
    source = Path(__file__).resolve()
    for parent in source.parents:
        candidate = parent / 'ML-for-IR-drop'
        if (candidate / 'benchmarks').is_dir():
            return candidate
    return source.parents[3] / 'ML-for-IR-drop'


DATA_ROOT = _find_benchmark_root()
BENCH = str(DATA_ROOT / 'benchmarks')
SOL = str(DATA_ROOT / 'solution')
