"""Global config for powerbox."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any
from typing import Generator

import logging
import toml

log = logging.getLogger(__name__)


class Config:
    """Class that handles the configuration file."""

    def __init__(self, config_file: str | Path | None = None) -> None:
        if config_file is None:
            config_file = Path().parent.absolute() / "config.toml"
        else:
            config_file = Path(config_file)

        self.config_file = config_file

        if not self.config_file.exists():
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            self.config_file.touch()
            log.info(f"Created new config file at {self.config_file}")

        self.config = toml.loads(self.config_file.read_text())

        # Set defaults
        if "USE_FFTW" not in self:
            self["USE_FFTW"] = True
        if "THREADS" not in self:
            self['THREADS'] = None


    def __getitem__(self, key: str) -> Any:
        """Get a value from the config file."""
        return self.config[key]

    def __setitem__(self, key: str, value: Any):
        """Set a value in the config file."""
        self.config[key] = value
        self.config_file.write_text(toml.dumps(self.config))

    def __delitem__(self, key: str) -> None:
        """Delete a value from the config file."""
        del self.config[key]
        self.config_file.write_text(toml.dumps(self.config))

    def __contains__(self, key: str) -> bool:
        """Check if a key is in the config file."""
        return key in self.config

    def __repr__(self) -> str:
        """Get the string representation of the config file."""
        return repr(self.config)

    def __str__(self) -> str:
        """Get the string representation of the config file."""
        return str(self.config)

    def keys(self) -> Generator[str, None, None]:
        """Yield the keys in the config file."""
        yield from self.config.keys()

    def values(self) -> Generator[Any, None, None]:
        """Yield the values in the config file."""
        yield from self.config.values()

    def items(self) -> Generator[tuple[str, Any], None, None]:
        """Yield the keys and values of the main data products, like a dict."""
        yield from self.config.items()

    def update(self, **kw) -> None:
        """Update the config file with new values."""
        self.config.update(**kw)
        self.config_file.write_text(toml.dumps(self.config))

    @contextmanager
    def use(self, **kw) -> Generator[None, None, None]:
        """Use configuration values temporarily."""
        old = {k: self[k] for k in kw}
        self.update(**kw)
        try:
            yield
        finally:
            self.update(**old)


CONFIG = Config()