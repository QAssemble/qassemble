"""Console entry point for launching a QAssemble calculation."""
from .run import Run


def main():
    """Run the command-line entry point."""
    print("Calculation Start")
    Run()
    print("Calculation Finish")


if __name__ == "__main__":
    main()
