import logging

from .Logger import setup_logger
from .Run import Run


def main():
    # Read logfile name from input.ini if available
    logfile = "stdout.log"
    try:
        loc = {}
        glob = {}
        exec(open("input.ini").read(), glob, loc)
        if "Control" in loc and "LogFile" in loc["Control"]:
            logfile = loc["Control"]["LogFile"]
    except FileNotFoundError:
        pass

    logger = setup_logger(logfile=logfile)
    logger.info("Calculation Start")
    Run()
    logger.info("Calculation Finish")


if __name__ == "__main__":
    main()
