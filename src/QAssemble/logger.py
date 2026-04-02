import logging
import sys


def setup_logger(logfile="stdout.log", level=logging.INFO):
    """Setup the QAssemble package-wide logger.

    Creates a logger named 'QAssemble' with two handlers:
    - FileHandler: writes to `logfile` (default: stdout.log)
    - StreamHandler: writes to sys.stdout (console)

    Parameters
    ----------
    logfile : str
        Path to the log file. Default is 'stdout.log'.
    level : int
        Logging level. Default is logging.INFO.

    Returns
    -------
    logging.Logger
        The configured logger instance.
    """
    logger = logging.getLogger("QAssemble")
    logger.setLevel(level)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(logfile, mode="w")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
