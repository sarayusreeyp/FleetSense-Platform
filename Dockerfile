# ==============================================================================
# FleetSense Production Dockerfile (Multi-stage)
# ==============================================================================

# Stage 1: Build & Dependencies
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


# Stage 2: Lean Runtime Image
FROM python:3.12-slim AS runtime

# Set non-buffering and bytecode suppression
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    FLEETSENSE_CONFIG_PATH="/app/configs/app.yaml"

WORKDIR /app

# Install runtime utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create unprivileged user for security compliance
RUN groupadd -g 10001 fleetsense && \
    useradd -u 10001 -g fleetsense -s /bin/bash -m appuser

# Copy application code and configurations
COPY configs/ /app/configs/
COPY src/ /app/src/
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md

# Install package in editable mode within virtual environment
RUN /opt/venv/bin/pip install --no-deps -e /app

# Create persistent storage directories with appropriate permissions
RUN mkdir -p /app/artifacts/models /app/artifacts/metadata /app/data /app/logs && \
    chown -R appuser:fleetsense /app

# Switch to unprivileged user
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Docker healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: launch Uvicorn production server
CMD ["uvicorn", "fleetsense.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
