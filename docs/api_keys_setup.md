# API Keys and Secrets Setup Guide

This guide describes how to configure external API credentials to unlock premium or higher-rate capabilities in the **FAST Stock Analysis WebApp**. 

Currently, the application supports Reddit API integration via **PRAW** (Python Reddit API Wrapper) for stock sentiment analysis. If no keys are provided, the app falls back to unauthenticated requests, which are subject to strict rate limits.

---

## 1. Obtain Reddit API Credentials

To query Reddit for sentiment data reliably, you need to register a developer application:

1. Log into your Reddit account.
2. Navigate to the **[Reddit App Preferences Dashboard](https://www.reddit.com/prefs/apps)**.
3. Scroll to the bottom and click **"are you a developer? create another app..."**.
4. Fill in the fields:
   * **Name**: `FAST Stock Analysis` (or any custom name)
   * **App Type**: Select the **script** radio button (important).
   * **Description**: Optional.
   * **About URL**: Optional.
   * **Redirect URI**: `http://localhost:8080` (or any valid URL).
5. Click **"create app"**.
6. Note your credentials:
   * **Client ID**: The alphanumeric string directly under the app name (e.g., `_aBc123XYZ_`).
   * **Client Secret**: The string labeled **secret** (e.g., `-dEf456UVW_xyz...`).

---

## 2. Local Configuration (Development)

Streamlit reads credentials from a local configuration file located at `.streamlit/secrets.toml` in your project root.

1. In the project root directory, create a folder named `.streamlit` (if it does not exist).
2. Inside `.streamlit`, create a file named `secrets.toml`.
3. Add your credentials in the following format:

```toml
[reddit]
client_id = "YOUR_REDDIT_CLIENT_ID"
client_secret = "YOUR_REDDIT_CLIENT_SECRET"
user_agent = "FAST-Stock-Analysis App (v1.0)"
```

> [!NOTE]
> `.streamlit/secrets.toml` is included in `.gitignore` by default to prevent accidentally committing private keys to public repositories.

---

## 3. Production Configuration (Streamlit Community Cloud)

If you deploy your app using the [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Go to your Streamlit Cloud Dashboard and click on your deployed app.
2. Open the **App settings** panel (gear icon).
3. Select **Secrets** on the left menu.
4. Paste the identical TOML code block in the editor:
   ```toml
   [reddit]
   client_id = "YOUR_REDDIT_CLIENT_ID"
   client_secret = "YOUR_REDDIT_CLIENT_SECRET"
   user_agent = "FAST-Stock-Analysis App (v1.0)"
   ```
5. Click **Save**. The application will automatically reload and inject the secrets.

---

## 4. Fallback Mode

If no secrets are defined in the workspace:
* The application catches the exception and falls back to making standard HTTP requests to the public Reddit search endpoint (`https://www.reddit.com/r/.../search.json`).
* While this allows the app to function without keys, public endpoints are rate-limited heavily. Under heavy usage or concurrent users, you may encounter `429 Too Many Requests` status codes, leading to empty or partial sentiment results.
