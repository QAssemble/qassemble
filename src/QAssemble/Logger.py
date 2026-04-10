import logging
import os
import sys


def setup_logger(logfile="stdout.log", level=logging.INFO, enabled=True):
    """Setup the QAssemble package-wide logger.

    Creates a logger named 'QAssemble' with separate handlers:
    - FileHandler (INFO): writes info/debug messages to `logfile`
    - FileHandler (WARNING): writes error/warning messages to `stderr.log`
    - StreamHandler: writes all messages to sys.stdout (console)

    Parameters
    ----------
    logfile : str
        Path to the log file. Default is 'stdout.log'.
    level : int
        Logging level. Default is logging.INFO.
    enabled : bool
        Whether to attach active handlers. When False, installs a NullHandler
        so non-root MPI ranks stay silent.

    Returns
    -------
    logging.Logger
        The configured logger instance.
    """
    logger = logging.getLogger("QAssemble")
    logger.setLevel(level)
    logger.propagate = False

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()

    if not enabled:
        logger.addHandler(logging.NullHandler())
        return logger

    info_formatter = logging.Formatter("%(message)s")
    error_formatter = logging.Formatter("%(levelname)s: %(message)s")

    # Info file handler — only INFO and DEBUG
    fh_info = logging.FileHandler(logfile, mode="w")
    fh_info.setLevel(level)
    fh_info.addFilter(lambda record: record.levelno < logging.WARNING)
    fh_info.setFormatter(info_formatter)
    logger.addHandler(fh_info)

    # Error file handler — WARNING and above
    errfile = os.path.join(os.path.dirname(logfile) or ".", "stderr.log")
    fh_err = logging.FileHandler(errfile, mode="w")
    fh_err.setLevel(logging.WARNING)
    fh_err.setFormatter(error_formatter)
    logger.addHandler(fh_err)

    # Console handler — all messages
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(info_formatter)
    logger.addHandler(ch)

    return logger
