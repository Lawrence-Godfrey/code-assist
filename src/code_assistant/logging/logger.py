import logging
from typing import Optional


class LoggingConfig:
    """Singleton configuration class for logging settings."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._enabled = True
        return cls._instance

    @property
    def enabled(self) -> bool:
        """Get the current logging state."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set the logging state."""
        self._enabled = bool(value)

    @classmethod
    def reset(cls) -> None:
        """Reset the configuration to default state (useful for testing)."""
        cls._instance = None


class LogFilter(logging.Filter):
    """Filter that checks LoggingConfig to determine if logs should be shown."""

    def filter(self, record):
        """Only allow logs if logging is enabled in config."""
        return LoggingConfig().enabled


class CodeAssistantLogger:
    """Custom logger for centralized logging control."""

    _instance: Optional[logging.Logger] = None

    @classmethod
    def get_logger(cls, name: Optional[str] = None) -> logging.Logger:
        """
        Get or create a logger instance with proper configuration.

        Args:
            name: Optional name for the logger. If None, uses the module name.

        Returns:
            Configured logger instance
        """
        if cls._instance is None:
            # Create and configure root logger
            logger = logging.getLogger("code_assistant")
            logger.setLevel(logging.INFO)

            # Prevent propagation to root logger
            logger.propagate = False

            # Create console handler if none exists
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setLevel(logging.INFO)

                # Create formatter
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                handler.setFormatter(formatter)

                # Add filter
                log_filter = LogFilter()
                handler.addFilter(log_filter)

                # Add handler to logger
                logger.addHandler(handler)

            cls._instance = logger

        if name:
            # Return a child logger if name is provided
            return cls._instance.getChild(name)

        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the logger (useful for testing)."""
        cls._instance = None


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Convenience function to get a logger instance.

    Args:
        name: Optional name for the logger. If None, uses the module name.

    Returns:
        Configured logger instance
    """
    return CodeAssistantLogger.get_logger(name)
