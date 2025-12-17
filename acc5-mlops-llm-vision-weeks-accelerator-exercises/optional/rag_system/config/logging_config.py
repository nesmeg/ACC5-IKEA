import logging
import logging.handlers
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import asyncio

class CustomFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with additional metadata.

        Args:
            record: Log record to format

        Returns:
            Formatted log string in JSON
        """
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "file": record.filename,
            "line": record.lineno,
            "process": record.process,
            "message": record.getMessage(),
        }

        # Add extra fields if available
        if hasattr(record, "extra"):
            log_data["extra"] = record.extra

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)

class StrategyContextFilter(logging.Filter):
    """Filter for adding strategy context to log records."""

    def __init__(self, strategy_name: str):
        """
        Initialize filter with strategy name.

        Args:
            strategy_name: Name of the strategy
        """
        super().__init__()
        self.strategy_name = strategy_name

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add strategy context to log record.

        Args:
            record: Log record to filter

        Returns:
            Boolean indicating whether to include the record
        """
        record.strategy = self.strategy_name
        return True

def setup_logging(
    log_dir: str = "logs",
    level: str = "INFO",
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Setup comprehensive logging configuration with separate formatters for console and files.

    Args:
        log_dir: Directory for log files
        level: Base logging level
        config: Additional configuration options
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Default configuration
    if config is None:
        config = {
            "file_logging": True,
            "console_logging": True,
            "log_rotation": True,
            "max_bytes": 10485760,  # 10MB
            "backup_count": 5,
        }

    # Define formatters
    json_formatter = CustomFormatter()
    simple_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Initialize handlers dictionary
    handlers = {}

    if config.get("file_logging", True):
        # Debug log file
        debug_handler = logging.handlers.RotatingFileHandler(
            log_path / f'debug_{timestamp}.log',
            maxBytes=config.get("max_bytes", 10485760),
            backupCount=config.get("backup_count", 5)
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(json_formatter)
        handlers['debug'] = debug_handler

        # Info log file
        info_handler = logging.handlers.RotatingFileHandler(
            log_path / f'info_{timestamp}.log',
            maxBytes=config.get("max_bytes", 10485760),
            backupCount=config.get("backup_count", 5)
        )
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(json_formatter)
        handlers['info'] = info_handler

        # Error log file
        error_handler = logging.handlers.RotatingFileHandler(
            log_path / f'error_{timestamp}.log',
            maxBytes=config.get("max_bytes", 10485760),
            backupCount=config.get("backup_count", 5)
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(json_formatter)
        handlers['error'] = error_handler

    if config.get("console_logging", True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        handlers['console'] = console_handler

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to prevent duplication
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Add new handlers
    for handler in handlers.values():
        root_logger.addHandler(handler)

    # Setup strategy loggers
    strategy_names = ["basic", "multi_query", "hypothetical", "step_back"]
    for strategy in strategy_names:
        strategy_logger = logging.getLogger(f"rag.strategy.{strategy}")
        strategy_logger.setLevel(logging.DEBUG)
        strategy_logger.addFilter(StrategyContextFilter(strategy))
        # Ensure strategy loggers propagate to root
        strategy_logger.propagate = True

    # Log initialization
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized", extra={
        "extra": {
            "config": {
                "log_dir": str(log_path),
                "level": level,
                "handlers": list(handlers.keys())
            }
        }
    })

def setup_specific_logger(
    logger_name: str,
    log_dir: str,
    level: int = logging.INFO,
    formatter: Optional[logging.Formatter] = None,
    handler_type: str = "file",
    **kwargs
) -> logging.Logger:
    """
    Generalized function to set up specific loggers.

    Args:
        logger_name: Name of the logger
        log_dir: Directory for log files
        level: Logging level
        formatter: Formatter to use
        handler_type: Type of handler ('file', 'rotating', 'timed')
        **kwargs: Additional arguments for handlers

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent logs from propagating to root

    log_path = Path(log_dir) / logger_name.split('.')[-1]
    log_path.mkdir(parents=True, exist_ok=True)

    if handler_type == "file":
        handler = logging.FileHandler(log_path / f"{logger_name.split('.')[-1]}.log")
    elif handler_type == "rotating":
        handler = logging.handlers.RotatingFileHandler(
            log_path / f"{logger_name.split('.')[-1]}.log",
            maxBytes=kwargs.get("max_bytes", 10485760),
            backupCount=kwargs.get("backup_count", 5)
        )
    elif handler_type == "timed":
        handler = logging.handlers.TimedRotatingFileHandler(
            log_path / f"{logger_name.split('.')[-1]}.log",
            when=kwargs.get("when", "midnight"),
            interval=kwargs.get("interval", 1),
            backupCount=kwargs.get("backup_count", 30)
        )
    else:
        raise ValueError("Unsupported handler type.")

    handler.setLevel(level)
    handler.setFormatter(formatter if formatter else CustomFormatter())
    logger.addHandler(handler)

    return logger

def setup_strategy_logger(strategy_name: str, log_dir: str) -> logging.Logger:
    """
    Setup logger for a specific strategy with dedicated handlers.

    Args:
        strategy_name: Name of the strategy
        log_dir: Directory for log files

    Returns:
        Configured logger instance
    """
    logger = setup_specific_logger(
        logger_name=f"rag.strategy.{strategy_name}",
        log_dir=log_dir,
        level=logging.DEBUG,
        handler_type="file"
    )
    logger.addFilter(StrategyContextFilter(strategy_name))
    return logger

def setup_query_logger(log_dir: str) -> logging.Logger:
    """
    Setup logger for query processing with rotation.

    Args:
        log_dir: Directory for log files

    Returns:
        Configured logger instance
    """
    logger = setup_specific_logger(
        logger_name="rag.query",
        log_dir=log_dir,
        level=logging.INFO,
        handler_type="rotating",
        max_bytes=10485760,  # 10MB
        backup_count=5
    )
    return logger

def setup_metrics_logger(log_dir: str) -> logging.Logger:
    """
    Setup logger for metrics collection.

    Args:
        log_dir: Directory for log files

    Returns:
        Configured logger instance
    """
    logger = setup_specific_logger(
        logger_name="rag.metrics",
        log_dir=log_dir,
        level=logging.INFO,
        handler_type="timed",
        when="midnight",
        interval=1,
        backup_count=30
    )
    return logger

def setup_error_logger(log_dir: str) -> logging.Logger:
    """
    Setup logger for error tracking and monitoring.

    Args:
        log_dir: Directory for log files

    Returns:
        Configured logger instance
    """
    logger = setup_specific_logger(
        logger_name="rag.errors",
        log_dir=log_dir,
        level=logging.ERROR,
        handler_type="file"
    )
    return logger

class LogManager:
    """Manager class for centralized logging configuration and access."""

    def __init__(self, log_dir: str = "logs"):
        """
        Initialize log manager.

        Args:
            log_dir: Base directory for logs
        """
        self.log_dir = Path(log_dir)
        self.loggers = {}

        # Setup base logging
        setup_logging(log_dir)

        # Initialize specialized loggers
        self.query_logger = setup_query_logger(log_dir)
        self.metrics_logger = setup_metrics_logger(log_dir)
        self.error_logger = setup_error_logger(log_dir)

        # Setup strategy loggers
        strategy_names = ["basic", "multi_query", "hypothetical", "step_back"]
        for strategy in strategy_names:
            self.loggers[strategy] = setup_strategy_logger(strategy, log_dir)

    def get_strategy_logger(self, strategy_name: str) -> logging.Logger:
        """
        Get logger for specific strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Strategy-specific logger
        """
        if strategy_name not in self.loggers:
            self.loggers[strategy_name] = setup_strategy_logger(strategy_name, self.log_dir)
        return self.loggers[strategy_name]

    def log_query(self, query: str, metadata: Dict[str, Any]) -> None:
        """
        Log query execution details.

        Args:
            query: Query string
            metadata: Query execution metadata
        """
        self.query_logger.info("Query execution", extra={
            "extra": {
                "query": query,
                **metadata
            }
        })

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Log system metrics.

        Args:
            metrics: Metrics dictionary
        """
        self.metrics_logger.info("System metrics", extra={
            "extra": {
                "metrics": metrics,
                "timestamp": datetime.now().isoformat()
            }
        })

    def log_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """
        Log error with context.

        Args:
            error: Exception instance
            context: Error context
        """
        self.error_logger.error(
            "Error occurred",
            exc_info=error,
            extra={"extra": {"context": context}}
        )

class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler for high-performance logging."""

    def __init__(self):
        """Initialize async log handler."""
        super().__init__()
        self.queue = asyncio.Queue()
        self.task = None
        self.running = False

    async def start(self):
        """Start async log processing."""
        self.running = True
        self.task = asyncio.create_task(self._process_logs())

    async def stop(self):
        """Stop async log processing."""
        self.running = False
        if self.task:
            await self.task

    def emit(self, record):
        """
        Emit log record to queue.

        Args:
            record: Log record to emit
        """
        if self.running:
            asyncio.create_task(self.queue.put(record))

    async def _process_logs(self):
        """Process logs from queue."""
        while self.running:
            try:
                record = await self.queue.get()
                await self._write_log(record)
            except Exception as e:
                print(f"Error processing log: {e}")

    async def _write_log(self, record):
        """
        Write log record.

        Args:
            record: Log record to write
        """
        try:
            msg = self.format(record)
            # Implement actual async writing logic here
            pass
        except Exception as e:
            print(f"Error writing log: {e}")

def get_logger(name: str) -> logging.Logger:
    """
    Get configured logger by name.

    Args:
        name: Logger name

    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"rag.{name}")
