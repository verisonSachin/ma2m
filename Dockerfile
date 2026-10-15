# Manga2Motion: AI-Powered Manga-to-Video Diffusion System
# Optimized Dockerfile with CUDA 12.1 runtime and Astral UV package manager

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive debconf prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy \
    PORT=7860 \
    HOST=0.0.0.0

# Install Linux system dependencies, FFmpeg, and media codecs
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    curl \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Astral UV binary directly from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency specifications first for optimal Docker layer caching
COPY pyproject.toml requirements.txt ./

# Install python dependencies with uv
RUN uv pip install --system --no-cache -r requirements.txt

# Create persistence directories for models, input pages, and output renders
RUN mkdir -p /app/models /app/inputs /app/outputs /app/config

# Copy application source code and scripts
COPY config/ /app/config/
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY plan.md /app/plan.md

# Make helper scripts executable
RUN chmod +x /app/scripts/*.sh || true

# Expose Gradio WebUI port
EXPOSE 7860

# Healthcheck to monitor WebUI availability
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Default execution: Launch interactive Gradio WebUI via uv
CMD ["uv", "run", "python3", "src/ui/app.py"]
