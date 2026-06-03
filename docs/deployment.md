# Deployment and Operations Guide

This guide outlines deployment options for the **FAST Stock Analysis WebApp**, covering local environment setup, Docker containerization, and Streamlit Community Cloud deployment.

---

## 1. Local Deployment

For running the application locally on your machine:

1. **Clone the repository and install dependencies**:
   ```bash
   git clone https://github.com/jayshilj/FAST-Stock-Analysis-WebApp.git
   cd FAST-Stock-Analysis-WebApp
   python -m venv venv
   source venv/bin/activate  # On Windows use: .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. **Launch Streamlit**:
   ```bash
   streamlit run app.py
   ```
3. Open your browser and navigate to `http://localhost:8501`.

---

## 2. Docker Containerization

To package the application as a standalone container, you can use the following Docker environment:

### Create a `Dockerfile`
Create a `Dockerfile` in the root of the project with the following contents:

```dockerfile
# Use a lightweight official Python runtime
FROM python:3.9-slim

# Avoid writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies needed for compiling python packages if any
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source files
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run the Image

1. **Build the Docker Image**:
   ```bash
   docker build -t fast-stock-analysis:latest .
   ```
2. **Run the Container**:
   ```bash
   docker run -d -p 8501:8501 --name fast-stock-analysis-app fast-stock-analysis:latest
   ```
3. Navigate to `http://localhost:8501` to test the running container.

---

## 3. Deploying to Streamlit Community Cloud

Streamlit provides free hosting for public GitHub repositories:

1. Push your code to your public GitHub repository (e.g., `origin/master`).
2. Log into **[Streamlit Share](https://share.streamlit.io)** using your GitHub credentials.
3. Click the **"New app"** button.
4. Fill in the deployment details:
   * **Repository**: `jayshilj/FAST-Stock-Analysis-WebApp`
   * **Branch**: `master`
   * **Main file path**: `app.py`
5. Click **"Deploy!"**. Your app will build, install the dependencies listed in `requirements.txt`, and become accessible via a public URL.
6. Remember to configure your API keys (like Reddit client IDs) in the Streamlit Cloud Dashboard secrets if you want to support live social sentiment analysis (see [API Keys Setup Guide](api_keys_setup.md)).

---

## 4. Production Performance Tuning

To optimize runtime efficiency in production:
* **Caching Settings**: The app makes extensive use of `@st.cache_data`. This caches ticker downloads (`yfinance`) and web scrapers (`FinViz`) to reduce execution time and avoid API rate limits.
* **Server Port/Address Mapping**: You can configure default host porting and headless settings by adding or editing options in [config.toml](file:///c:/Users/jaysh/OpenSourceContributions/FAST-Stock-Analysis-WebApp/.streamlit/config.toml):
  ```toml
  [server]
  headless = true
  port = 8501
  enableCORS = false
  ```
