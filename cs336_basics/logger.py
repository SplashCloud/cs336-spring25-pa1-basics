"""
LoggerManager: A comprehensive logging system for the CS336 project.

This module provides a LoggerManager class that can be instantiated and used
throughout the project to log messages with different levels (DEBUG, INFO, WARNING, ERROR),
including timestamp, location information (file, line, function), and optional file output.
"""

import os
import sys
import inspect
from datetime import datetime
from enum import IntEnum
from typing import Optional, TextIO
from pathlib import Path


class LogLevel(IntEnum):
    """Log levels in order of severity."""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3


class LoggerManager:
    """
    A comprehensive logger manager for the CS336 project.
    
    Features:
    - Multiple log levels (DEBUG, INFO, WARNING, ERROR)
    - Timestamp and location information (file, line, function)
    - Console and file output support
    - Configurable log level filtering
    - Formatted output with colors (optional)
    """
    
    # ANSI color codes for terminal output
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'RESET': '\033[0m'        # Reset
    }
    
    def __init__(
        self,
        name: str = "Logger",
        log_level: LogLevel = LogLevel.INFO,
        log_to_file: bool = False,
        log_file_path: Optional[str] = None,
        log_dir: str = "logs",
        use_colors: bool = True,
        include_location: bool = True
    ):
        """
        Initialize the LoggerManager.
        
        Args:
            name: Name identifier for this logger instance
            log_level: Minimum log level to output (default: INFO)
            log_to_file: Whether to write logs to a file (default: False)
            log_file_path: Path to log file. If None, auto-generates based on name (default: None)
            log_dir: Directory for log files (default: "logs")
            use_colors: Whether to use colors in console output (default: True)
            include_location: Whether to include file/line/function info (default: True)
        """
        self.name = name
        self.log_level = log_level
        self.use_colors = use_colors and sys.stdout.isatty()  # Only use colors if terminal supports it
        self.include_location = include_location
        self.log_file: Optional[TextIO] = None
        
        if log_to_file:
            # Create log directory if it doesn't exist
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)
            
            # Generate log file path if not provided
            if log_file_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file_path = log_dir_path / f"{name}_{timestamp}.log"
            else:
                log_file_path = Path(log_file_path)
            
            # Open log file in append mode
            self.log_file = open(log_file_path, 'w', encoding='utf-8')
    
    def _get_caller_info(self) -> tuple[str, int, str]:
        """Get information about the caller (file, line, function)."""
        # Get the frame of the caller (skip _log, info/debug/warning/error methods)
        frame = inspect.currentframe()
        try:
            # Go up the call stack to find the actual caller
            caller_frame = frame.f_back.f_back.f_back.f_back
            filename = os.path.basename(caller_frame.f_code.co_filename)
            line_no = caller_frame.f_lineno
            function = caller_frame.f_code.co_name
            return filename, line_no, function
        finally:
            del frame
    
    def _format_message(
        self,
        level: str,
        message: str,
        include_location: Optional[bool] = None
    ) -> str:
        """
        Format a log message with timestamp, level, location, and message.
        
        Args:
            level: Log level string (DEBUG, INFO, WARNING, ERROR)
            message: The log message
            include_location: Override default include_location setting
            
        Returns:
            Formatted log message string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Include milliseconds
        
        include_loc = include_location if include_location is not None else self.include_location
        
        if include_loc:
            filename, line_no, function = self._get_caller_info()
            location_str = f"{filename}:{line_no}:{function}"
            formatted = f"[{timestamp}] [{level:7s}] [{self.name}] [{location_str}] {message}"
        else:
            formatted = f"[{timestamp}] [{level:7s}] [{self.name}] {message}"
        
        return formatted
    
    def _log(
        self,
        level: LogLevel,
        level_str: str,
        message: str,
        include_location: Optional[bool] = None
    ):
        """
        Internal method to log a message.
        
        Args:
            level: Log level enum value
            level_str: Log level string
            message: The log message
            include_location: Override default include_location setting
        """
        if level < self.log_level:
            return  # Skip if below minimum log level
        
        formatted_message = self._format_message(level_str, message, include_location)
        
        if self.log_file is None: # Output to console
            if self.use_colors:
                color = self.COLORS.get(level_str, self.COLORS['RESET'])
                reset = self.COLORS['RESET']
                print(f"{color}{formatted_message}{reset}", flush=True)
            else:
                print(formatted_message, flush=True)
        else: # Output to file (without colors)
            self.log_file.write(formatted_message + '\n')
            self.log_file.flush()
    
    def debug(self, message: str, include_location: Optional[bool] = None):
        """Log a DEBUG level message."""
        self._log(LogLevel.DEBUG, "DEBUG", message, include_location)
    
    def info(self, message: str, include_location: Optional[bool] = None):
        """Log an INFO level message."""
        self._log(LogLevel.INFO, "INFO", message, include_location)
    
    def warning(self, message: str, include_location: Optional[bool] = None):
        """Log a WARNING level message."""
        self._log(LogLevel.WARNING, "WARNING", message, include_location)
    
    def error(self, message: str, include_location: Optional[bool] = None):
        """Log an ERROR level message."""
        self._log(LogLevel.ERROR, "ERROR", message, include_location)
    
    def log_loss(self, epoch: int, iteration: int, loss: float, include_location: Optional[bool] = None):
        """
        Special method for logging training loss.
        
        Args:
            iteration: Current training iteration
            loss: Loss value
            include_location: Override default include_location setting
        """
        message = f"loss={loss:.6f} at iteration {iteration} at epoch {epoch}"
        self.info(message, include_location)
    
    def log_metric(self, name: str, value: float, iteration: Optional[int] = None, include_location: Optional[bool] = None):
        """
        Log a training metric.
        
        Args:
            name: Name of the metric
            value: Metric value
            iteration: Optional iteration number
            include_location: Override default include_location setting
        """
        if iteration is not None:
            message = f"{name}={value:.6f} at iteration {iteration}"
        else:
            message = f"{name}={value:.6f}"
        self.info(message, include_location)
    
    def set_log_level(self, level: LogLevel):
        """Change the minimum log level."""
        self.log_level = level
    
    def close(self):
        """Close the log file if it's open."""
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures log file is closed."""
        self.close()
        return False
    
    def __del__(self):
        """Destructor - ensures log file is closed."""
        self.close()

