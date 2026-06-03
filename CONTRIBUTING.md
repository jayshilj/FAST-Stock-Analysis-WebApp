# Contributing to FAST Stock Analysis WebApp

Thank you for your interest in contributing to the **FAST Stock Analysis WebApp**! This document outlines guidelines and best practices for setting up your environment, submitting pull requests, and adhering to our code standards.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please report any unacceptable behavior to the project maintainers.

## Getting Started

### Prerequisites

Make sure you have the following installed on your machine:
* Python 3.8 or higher
* Git

### Local Development Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/jayshilj/FAST-Stock-Analysis-WebApp.git
   cd FAST-Stock-Analysis-WebApp
   ```

2. **Create a Virtual Environment**
   On Windows:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
   On macOS/Linux:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## Development Workflow

### Git Branching Model

We follow a simple branch-per-feature/bugfix workflow.
* Create branches off `master` with descriptive names:
  * Features: `feature/your-feature-name`
  * Bug fixes: `bugfix/issue-description`
  * Documentation: `docs/what-is-changing`

### Commit Message Guidelines

We use [Conventional Commits](https://www.conventionalcommits.org/) to structure our commit history. Commits should use the format:
```
<type>(<scope>): <short description>
```

Types allowed:
* `feat`: A new feature
* `fix`: A bug fix
* `docs`: Documentation changes
* `style`: Code formatting changes (whitespace, missing semi-colons, etc.)
* `refactor`: Code changes that neither fix a bug nor add a feature
* `test`: Adding or updating tests
* `chore`: Updates to build scripts, dependencies, etc.

*Example:* `feat: add relative strength index (RSI) technical indicator`

### Running Tests

Before submitting a pull request, run the test suite to ensure all unit tests pass:
```bash
pytest
```

## Pull Request Guidelines

1. Ensure all code conforms to Python PEP 8 style standards.
2. Add unit tests for any new modules or mathematical calculations (e.g., indicators in `indicators.py`).
3. Update the `CHANGELOG.md` with brief notes of your changes.
4. Open a Pull Request pointing to `master` and provide a clear description of the modifications made.
