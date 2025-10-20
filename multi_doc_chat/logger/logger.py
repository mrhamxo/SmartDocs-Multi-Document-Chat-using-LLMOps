import os
import logging
from datetime import datetime
import structlog


class CustomLogger:
    def __init__(self, log_dir="logs"):
        # Create (or locate) the directory where log files will be stored
        self.logs_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Generate a log file name with a timestamp (e.g., 10_15_2025_17_25_30.log)
        log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
        self.log_file_path = os.path.join(self.logs_dir, log_file)

    def get_logger(self, name=__file__):
        # Use the file name as the logger's identifier
        logger_name = os.path.basename(name)

        # --- File Handler ---
        # Writes log messages to a log file
        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(message)s"))

        # --- Console Handler ---
        # Displays log messages in the terminal
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))

        # --- Configure the root Python logger ---
        # Ensures logs are output to both console and file
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[console_handler, file_handler]
        )

        # --- Configure structlog for structured JSON logging ---
        structlog.configure(
            processors=[
                # Adds ISO-format timestamps (UTC)
                structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
                # Includes the log level (INFO, ERROR, etc.)
                structlog.processors.add_log_level,
                # Renames the main log message field to "event"
                structlog.processors.EventRenamer(to="event"),
                # Renders the final log output as JSON
                structlog.processors.JSONRenderer()
            ],
            # Uses Python's standard logging system underneath
            logger_factory=structlog.stdlib.LoggerFactory(),
            # Improves performance by caching the logger
            cache_logger_on_first_use=True,
        )

        # Return a structured logger instance you can use
        return structlog.get_logger(logger_name)
