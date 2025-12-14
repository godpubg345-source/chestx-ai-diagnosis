# Multi-stage Dockerfile for ChestX-AI
# Optimized for GPU inference

# Stage 1: Base image with CUDA
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Stage 2: Dependencies
FROM base AS dependencies

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Stage 3: Production
FROM dependencies AS production

# Copy application code
COPY app/ ./app/
COPY ml/ ./ml/
COPY models/ ./models/

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
