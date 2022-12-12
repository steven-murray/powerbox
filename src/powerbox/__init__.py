"""A package for creating mocks from input isotropic power spectra."""
try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

try:
    from ._version import version as __version__
except ModuleNotFoundError:  # pragma: no cover
    try:
        __version__ = version("powerbox")
    except PackageNotFoundError:
        # package is not installed
        __version__ = "unknown"

from .dft import fft, fftfreq, ifft
from .powerbox import LogNormalPowerBox, PowerBox
from .tools import angular_average, angular_average_nd, get_power
