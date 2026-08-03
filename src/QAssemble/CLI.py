"""Console entry point for launching a QAssemble calculation."""
import argparse

from .Run import Run


def main(argv=None):
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(
        prog="qassemble",
        description="Run a QAssemble calculation from a declarative input file.",
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        help="QAssemble input file to read. Defaults to qassemble.in.",
    )
    args = parser.parse_args(argv)

    print("Calculation Start")
    Run(input_file=args.input_file)
    print("Calculation Finish")


if __name__ == "__main__":
    main()
