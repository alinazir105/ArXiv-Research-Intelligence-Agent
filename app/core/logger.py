import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(name: str) -> logging.Logger:
    """Create and configure a logger for the given module name."""
    
    # getLogger returns the same logger instance if called with the same name —
    # this means calling setup_logger(__name__) in multiple files won't create
    # duplicate loggers. Each module gets its own named logger.
    logger = logging.getLogger(name)

    # if this logger already has handlers attached, it's already been configured —
    # return it as-is to avoid adding duplicate handlers on repeated calls.
    # this happens when a module is imported multiple times.
    if logger.handlers:
        return logger

    # DEBUG is the lowest level — setting it here means the logger accepts all
    # messages. Individual handlers then filter to their own minimum level.
    # without this, messages below WARNING would be silently dropped.
    logger.setLevel(logging.DEBUG)

    # formatter defines what each log line looks like:
    # 2026-01-15 14:23:01 | app.retrieval.retriever | INFO | Retrieval failed: ...
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # console handler — prints to terminal during development.
    # set to INFO so debug noise doesn't flood your terminal.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # create logs/ directory if it doesn't exist —
    # the file handler will fail if the directory isn't there.
    os.makedirs("logs", exist_ok=True)
    
    # RotatingFileHandler writes to logs/app.log.
    # when the file hits 10MB it renames it to app.log.1 and starts a fresh app.log.
    # backupCount=5 means it keeps app.log + app.log.1 through app.log.5 — 50MB total.
    # older backups are deleted automatically. you never fill up your disk.
    # set to DEBUG so every detail is captured in the file even if not shown in terminal.
    file_handler = RotatingFileHandler(
        "logs/app.log",
        maxBytes=10_000_000,  # 10MB per file
        backupCount=5         # keep last 5 rotated files
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # attach both handlers to the logger —
    # every log call now goes to both terminal and file simultaneously.
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger