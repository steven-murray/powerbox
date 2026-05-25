"""Generate backend benchmark plots for ``powerbox`` and ``powerbox.jax``.

This script benchmarks two operations:

1. Generating Gaussian real-space fields with :class:`powerbox.PowerBox` /
   :class:`powerbox.jax.PowerBox`.
2. Computing fully averaged power spectra with :func:`powerbox.get_power` /
   :func:`powerbox.jax.get_power`.

It writes:

- ``docs/_static/backend_benchmark_generation.png``
- ``docs/_static/backend_benchmark_power.png``
- ``docs/_static/backend_benchmark_results.json``

The benchmarks are intended for documentation and developer profiling, not as a
CI-enforced performance test.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import jax
import matplotlib
import numpy as np

jax.config.update("jax_enable_x64", True)
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import powerbox.jax as jpb  # noqa: E402
from powerbox import PowerBox, get_power  # noqa: E402
from powerbox.dft_backend import FFTW  # noqa: E402

DOCS_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = DOCS_DIR / "_static"
STATIC_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")

GENERATION_FIGURE = STATIC_DIR / "backend_benchmark_generation.png"
POWER_FIGURE = STATIC_DIR / "backend_benchmark_power.png"
RESULTS_JSON = STATIC_DIR / "backend_benchmark_results.json"

BOXLENGTH = 100.0
REPEATS = 2
WARMUPS = 1
FFTW_THREADS = 1
SIZES: dict[int, list[int]] = {
    1: [4096, 16384, 65536, 262144],
    2: [64, 128, 256, 512],
    3: [16, 32, 64, 96],
}


def pkfunc(k: np.ndarray | jax.Array) -> np.ndarray | jax.Array:
    """Return the reference power spectrum used for benchmarks."""
    return (1.0 + k) ** -2.0


def _gpu_name() -> str | None:
    """Return the first NVIDIA GPU name, if available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


def _time_numpy(fn: Callable[[], Any]) -> float:
    """Return the median runtime of a NumPy/FFTW benchmark callable."""
    for _ in range(WARMUPS):
        fn()

    timings: list[float] = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return statistics.median(timings)


def _time_jax(fn: Callable[[], Any]) -> float:
    """Return the median runtime of a JAX benchmark callable."""
    for _ in range(WARMUPS):
        result = fn()
        jax.block_until_ready(result)

    timings: list[float] = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        result = fn()
        jax.block_until_ready(result)
        timings.append(time.perf_counter() - start)
    return statistics.median(timings)


def _benchmark_numpy_generation(dim: int, n: int) -> float:
    """Benchmark Gaussian field generation with the NumPy FFT backend."""
    pb = PowerBox(
        n,
        dim=dim,
        pk=pkfunc,
        boxlength=BOXLENGTH,
        seed=1234,
        nthreads=1,
    )
    return _time_numpy(pb.delta_x)


def _benchmark_fftw_generation(dim: int, n: int) -> float:
    """Benchmark Gaussian field generation with the FFTW backend."""
    pb = PowerBox(
        n,
        dim=dim,
        pk=pkfunc,
        boxlength=BOXLENGTH,
        seed=1234,
        nthreads=FFTW_THREADS,
    )
    return _time_numpy(pb.delta_x)


def _benchmark_jax_generation(dim: int, n: int, device_kind: str) -> float:
    """Benchmark Gaussian field generation with JAX on the requested device."""
    device = jax.devices(device_kind)[0]
    keys = jax.random.split(jax.random.key(1234), WARMUPS + REPEATS + 1)

    with jax.default_device(device):
        pb = jpb.PowerBox(
            n,
            dim=dim,
            pk=pkfunc,
            boxlength=BOXLENGTH,
            key=keys[0],
        )

        key_iter = iter(keys[1:])

        def run() -> jax.Array:
            return pb.delta_x(key=next(key_iter))

        return _time_jax(run)


def _benchmark_numpy_power(dim: int, n: int) -> float:
    """Benchmark fully averaged power-spectrum estimation with NumPy FFT."""
    field = PowerBox(
        n,
        dim=dim,
        pk=pkfunc,
        boxlength=BOXLENGTH,
        seed=2024,
        nthreads=1,
    ).delta_x()

    def run() -> np.ndarray:
        return get_power(field, BOXLENGTH, nthreads=1, bins_upto_boxlen=True).power

    return _time_numpy(run)


def _benchmark_fftw_power(dim: int, n: int) -> float:
    """Benchmark fully averaged power-spectrum estimation with FFTW."""
    field = PowerBox(
        n,
        dim=dim,
        pk=pkfunc,
        boxlength=BOXLENGTH,
        seed=2024,
        nthreads=1,
    ).delta_x()

    def run() -> np.ndarray:
        return get_power(field, BOXLENGTH, nthreads=FFTW_THREADS, bins_upto_boxlen=True).power

    return _time_numpy(run)


def _benchmark_jax_power(dim: int, n: int, device_kind: str) -> float:
    """Benchmark fully averaged power-spectrum estimation with JAX."""
    device = jax.devices(device_kind)[0]

    with jax.default_device(device):
        field = jpb.PowerBox(
            n,
            dim=dim,
            pk=pkfunc,
            boxlength=BOXLENGTH,
            key=jax.random.key(2024),
        ).delta_x()

        def run() -> jax.Array:
            return jpb.get_power(field, BOXLENGTH, bins_upto_boxlen=True).power

        return _time_jax(run)


def _collect_results() -> list[dict[str, Any]]:
    """Collect benchmark results for all configured backends and dimensions."""
    results: list[dict[str, Any]] = []

    have_fftw = True
    try:
        FFTW(nthreads=FFTW_THREADS)
    except ImportError:
        have_fftw = False

    gpu_available = bool(jax.devices("gpu"))
    backends: list[tuple[str, Callable[[int, int], float], Callable[[int, int], float]]] = [
        ("numpy", _benchmark_numpy_generation, _benchmark_numpy_power),
        (
            "jax-cpu",
            lambda dim, n: _benchmark_jax_generation(dim, n, "cpu"),
            lambda dim, n: _benchmark_jax_power(dim, n, "cpu"),
        ),
    ]

    if have_fftw:
        backends.insert(1, ("fftw", _benchmark_fftw_generation, _benchmark_fftw_power))

    if gpu_available:
        backends.append(
            (
                "jax-gpu",
                lambda dim, n: _benchmark_jax_generation(dim, n, "gpu"),
                lambda dim, n: _benchmark_jax_power(dim, n, "gpu"),
            )
        )

    for dim, sizes in SIZES.items():
        for n in sizes:
            for backend_name, generation_bench, power_bench in backends:
                logging.info("benchmark generation backend=%s dim=%s n=%s", backend_name, dim, n)
                results.append(
                    {
                        "operation": "gaussian-field-generation",
                        "backend": backend_name,
                        "dim": dim,
                        "n_per_axis": n,
                        "seconds": generation_bench(dim, n),
                    }
                )
                logging.info("benchmark power backend=%s dim=%s n=%s", backend_name, dim, n)
                results.append(
                    {
                        "operation": "fully-averaged-power",
                        "backend": backend_name,
                        "dim": dim,
                        "n_per_axis": n,
                        "seconds": power_bench(dim, n),
                    }
                )

    return results


def _plot_results(results: list[dict[str, Any]], operation: str, output: Path) -> None:
    """Plot one benchmark operation with per-dimension subplots."""
    fig, axes = plt.subplots(1, len(SIZES), figsize=(15, 4.5), constrained_layout=True)
    backend_order = ["numpy", "fftw", "jax-cpu", "jax-gpu"]
    colors = {
        "numpy": "#1f77b4",
        "fftw": "#ff7f0e",
        "jax-cpu": "#2ca02c",
        "jax-gpu": "#d62728",
    }

    for ax, (dim, sizes) in zip(axes, SIZES.items(), strict=True):
        for backend in backend_order:
            subset = [
                row
                for row in results
                if row["operation"] == operation and row["backend"] == backend and row["dim"] == dim
            ]
            if not subset:
                continue

            subset = sorted(subset, key=lambda row: row["n_per_axis"])
            ax.loglog(
                [row["n_per_axis"] for row in subset],
                [row["seconds"] for row in subset],
                marker="o",
                label=backend,
                color=colors[backend],
            )

        ax.set_title(f"{dim}D")
        ax.set_xlabel("N per axis")
        ax.set_ylabel("Seconds")
        ax.set_xticks(sizes, [str(size) for size in sizes])
        ax.grid(True, which="both", alpha=0.3)

    axes[0].legend(frameon=False)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    """Run the benchmarks and write plots plus metadata."""
    results = _collect_results()

    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "fftw_threads": FFTW_THREADS,
        "fftw_interface_cache": True,
        "jax_version": jax.__version__,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_devices": [str(device) for device in jax.devices()],
        "gpu_name": _gpu_name(),
        "repeats": REPEATS,
        "warmups": WARMUPS,
        "boxlength": BOXLENGTH,
        "sizes": SIZES,
    }

    RESULTS_JSON.write_text(
        json.dumps({"metadata": metadata, "results": results}, indent=2),
        encoding="utf8",
    )
    _plot_results(results, "gaussian-field-generation", GENERATION_FIGURE)
    _plot_results(results, "fully-averaged-power", POWER_FIGURE)


if __name__ == "__main__":
    main()
