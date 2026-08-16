import logging

from rich.console import Console
from rich.logging import RichHandler


class Logger:
    """Singleton Rich-backed logger for Versal."""

    _logger: logging.Logger | None = None
    _console: Console | None = None

    @classmethod
    def get_logger(cls, level: int = logging.WARNING) -> logging.Logger:
        """Get the singleton logger instance."""
        if cls._logger is None:
            cls._logger = logging.getLogger("versal")
            cls._logger.setLevel(level)
            # Share the singleton console: a Live status footer can only hoist prints above its
            # pinned region when every writer goes through the ONE console it manages.
            handler = RichHandler(rich_tracebacks=True, console=cls.get_console(), show_path=False)
            handler.setLevel(level)
            handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
            cls._logger.addHandler(handler)
            cls._logger.propagate = False
        return cls._logger

    @classmethod
    def configure(cls, *, verbose: bool = False) -> logging.Logger:
        """Select clean default logging or concise operational diagnostics."""

        level = logging.INFO if verbose else logging.WARNING
        logger = cls.get_logger(level)
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)
        return logger

    @classmethod
    def get_console(cls) -> Console:
        """Get the singleton Rich console instance."""
        if cls._console is None:
            cls._console = Console()
        return cls._console
