# Security Policy

## Supported Versions

Currently, security updates and patches are actively provided for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| >= 1.0  | :white_check_mark: |
| < 1.0   | :x:                |

We recommend that all users run the latest version on the `master` branch.

## Reporting a Vulnerability

We take the security of this project seriously. If you find a security vulnerability, please do **not** report it via public GitHub issues. Instead, please report it following the process below:

1. **Send an Email**: Send a detailed security report to [insert-security-email-or-use-github-private-disclosure].
2. **Provide Details**: In your report, please include:
   * A detailed description of the vulnerability and its potential impact.
   * Step-by-step instructions (with a Proof of Concept / PoC) to reproduce the vulnerability.
   * Details about the operating system and environment (Python versions, browser versions) where it was tested.
3. **Response Timeline**: A maintainer will acknowledge receipt of your report within 48 hours and provide a timeline for addressing the issue.
4. **Coordinated Disclosure**: We ask you to follow coordinated vulnerability disclosure principles. Please do not disclose the vulnerability publicly or to third parties until we have had reasonable time to investigate, address, and deploy a fix.

## Security Practices

* **API Key Safety**: Never commit API keys or personal access tokens (such as Reddit API credentials or Twitter/StockTwits secrets) to the source code repository. Use Streamlit secrets (`.streamlit/secrets.toml`) or system environment variables.
* **Dependency Auditing**: We regularly audit our requirements (`requirements.txt`) to patch any underlying package vulnerabilities.
