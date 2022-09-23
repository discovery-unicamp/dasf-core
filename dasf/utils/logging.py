#!/usr/bin/env python3

from logging import INFO, Formatter, Logger, StreamHandler, getLogger


def init_logging() -> Logger:
    logger = getLogger("DASF")

    logger.setLevel(INFO)
    handler = StreamHandler()

    formatter = Formatter(

        fmt="[%(asctime)s] %(levelname)s - %(message)s",

        datefmt="%Y-%m-%d %H:%M:%S%z",

    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

