# Testing Strategy and Developer Guide

This document describes the testing architecture, test suites, and guidelines for running and writing tests in the **FAST Stock Analysis WebApp** repository.

---

## 1. Testing Philosophy

To keep the application robust and reliable, core logic is decoupled from Streamlit's runtime elements:
* **Business & Mathematical Logic** (e.g., technical indicator functions in `indicators.py`) are pure, stateless Python functions. This allows them to be unit-tested thoroughly without mocking browser sessions or Streamlit context.
* **Utility & Formatting Helpers** are isolated so they can be validated using standard inputs and outputs.

---

## 2. Test Directory Structure

The `tests/` directory contains all test suites and configurations:

* **[conftest.py](file:///c:/Users/jaysh/OpenSourceContributions/FAST-Stock-Analysis-WebApp/tests/conftest.py)**: Configures custom pytest markers (e.g., `@pytest.mark.slow` for intensive or api-calling tests).
* **[test_indicators.py](file:///c:/Users/jaysh/OpenSourceContributions/FAST-Stock-Analysis-WebApp/tests/test_indicators.py)**: Focuses strictly on verifying the mathematical correctness of calculations in `indicators.py`. Contains checks for:
  * Shape conformity (correct series length, preservation of index).
  * Standard parameters vs. custom parameter variations.
  * Edge cases (flat prices, NaN values, zero volume).
* **[test_helpers.py](file:///c:/Users/jaysh/OpenSourceContributions/FAST-Stock-Analysis-WebApp/tests/test_helpers.py)**: Validates string parsing, sentiment scoring wrappers, financial data formatting, and table cleansing.

---

## 3. How to Run Tests

### Prerequisites

Ensure you have installed testing dependencies. You can install them by running:
```bash
pip install pytest pytest-cov
```

### Execution Commands

Run all tests from the repository root:
```bash
pytest
```

Run tests with extra verbose output and short traceback display:
```bash
pytest -v
```

Filter and run only indicator tests:
```bash
pytest tests/test_indicators.py
```

Exclude slow-running integration tests:
```bash
pytest -m "not slow"
```

### Checking Test Coverage

To inspect what percentage of source files are covered by your unit tests:
```bash
pytest --cov=indicators --cov=ui_theme --cov-report=term-missing
```

---

## 4. Guidelines for Writing New Tests

When adding features or fixing bugs:

1. **Keep Functions Pure**: If you are introducing a new technical indicator or data transformation, place it in `indicators.py` or a dedicated helper file, keeping it free of Streamlit `st.*` UI calls.
2. **Add a Test Case**: Add a corresponding test function in `tests/test_*.py` prefixed with `test_`.
3. **Use Mocking for Network/I/O**: When writing tests that fetch data, use `unittest.mock` or `requests-mock` to avoid connecting to live stock/news feeds, ensuring speed and deterministic behavior.
