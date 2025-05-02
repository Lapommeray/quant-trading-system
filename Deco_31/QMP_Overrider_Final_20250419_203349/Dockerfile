# QMP Overrider Beyond God Mode - Dockerfile
# This Dockerfile creates a container for running the QMP Overrider system

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data config

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Expose ports for dashboard and API
EXPOSE 8501 8000

# Set entrypoint
ENTRYPOINT ["python", "main.py"]

# Default command (can be overridden)
CMD ["--mode", "dashboard"]
