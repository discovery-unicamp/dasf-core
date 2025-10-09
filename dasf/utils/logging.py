#!/usr/bin/env python3

""" Logging helpers for functions. """

import sys
from logging import INFO, Formatter, Logger, StreamHandler, getLogger


def init_logging() -> Logger:
    """
    Initialize logger objects to be used by modules.
    """
    logger = getLogger("DASF")

    logger.setLevel(INFO)
    handler = StreamHandler(sys.stdout)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = Formatter(
        fmt="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S%z",
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
