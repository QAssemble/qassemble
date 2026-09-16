# Contributing to QAssemble

Thank you for helping improve QAssemble. Bug reports, documentation fixes,
tests, and focused code changes are welcome.

## Issues

Search the existing issues before opening a new one. A useful bug report
includes:

- the QAssemble and Python versions;
- the operating system and installation method;
- a minimal `qassemble.in`, when relevant;
- the expected and observed behavior; and
- the complete traceback or error message.

For a new numerical method or a change to a public interface, open an issue
first so its physical conventions and compatibility can be agreed on before
implementation.

## Development setup

Clone the repository and install it with the test dependencies:

```bash
git clone https://github.com/QAssemble/qassemble.git
cd qassemble
python -m pip install -e ".[test]"
```

## Changes and tests

Keep each pull request focused. Add or update documentation when public
behavior changes. Numerical changes should state the relevant sign, index,
axis-ordering, and normalization conventions and include the smallest useful
regression test against a direct calculation or a documented reference value.

Run the regular test suite before submitting:

```bash
python -m pytest -q -m "not slow"
```

Run the checks relevant to the files you changed:

```bash
# Manuscript-scale graphene reproduction
python -m pytest -q -m slow

# Interactive examples
python -m pytest -q --nbmake examples/graphene/interactive/*.ipynb

# Documentation
python -m pip install mkdocs mkdocs-material
mkdocs build --strict
```

Do not commit generated HDF5 output, plots, caches, credentials, or private
input data. If a change affects stored HDF5 data or restart behavior, describe
the compatibility impact in the pull request.

## Pull requests

In the pull request description, explain the problem, the resulting behavior,
and the validation performed. Link the relevant issue when one exists. By
submitting a contribution, you agree that it is licensed under the repository's
GPL-3.0-or-later license.
