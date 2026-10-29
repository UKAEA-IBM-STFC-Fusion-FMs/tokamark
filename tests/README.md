# Tests

This directory contains the automated test suite for the code base, run with `pytest`:

```bash
pytest tests/
```

Individual test files can also still be run directly (e.g. `python tests/test_storage.py`), and via

```bash
python run_tests.py
```

which executes every `.py` file in the `tests/` directory (this one), excluding `run_tests.py` from recursion, and
reports pass/fail per file based on its exit code.

To add a test to the suite, add a `test_*.py` file to the `tests/` directory.
