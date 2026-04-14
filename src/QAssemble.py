#!/usr/bin/env python3
"""Backward-compatible entry point. Prefer `qassemble` or `python -m QAssemble`."""
from QAssemble.run import Run

if __name__ == "__main__":
    print("Calculation Start")
    Run()
    print("Calculation Finish")
