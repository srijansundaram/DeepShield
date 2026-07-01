# DeepShield — Production Docker Image
# Runs Streamlit app + FastAPI on a single container

FROM python:3.11-slim

# System deps — ffmpeg for AV sync, libGL for OpenCV
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (layer caching — only rebuilds if requirements change)
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn python-multipart librosa

# Copy project files
COPY . .

# Checkpoints must be mounted at runtime — not baked into the image
# (they are too large and change frequently)
RUN mkdir -p checkpoints

# Streamlit config — disable telemetry, set port
RUN mkdir -p ~/.streamlit && echo "\
[server]\n\
port = 8501\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n" > ~/.streamlit/config.toml

# Expose both ports
EXPOSE 8501 8000

# Startup script runs both services
CMD ["bash", "docker_start.sh"]