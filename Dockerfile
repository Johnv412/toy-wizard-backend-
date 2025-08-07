FROM python:3.11-slim

# Metadata
LABEL maintainer="ToyResaleWizard Security Team"
LABEL version="2.0"
LABEL description="Secure FastAPI backend for ToyResaleWizard"

# Security: Update package lists and install only necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgeos-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/archives/*

# Create non-root user early for security
RUN groupadd --gid 1000 toyresale && \
    useradd --uid 1000 --gid 1000 --create-home --no-log-init --shell /bin/false toyresale

# Set working directory
WORKDIR /app

# Security: Set proper ownership
RUN chown toyresale:toyresale /app

# Copy requirements first for better caching
COPY --chown=toyresale:toyresale requirements.txt .

# Install Python dependencies as root, then clean up
RUN pip install --no-cache-dir --upgrade pip==23.3.1 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge && \
    find /usr/local/lib/python* -name "*.pyc" -delete && \
    find /usr/local/lib/python* -name "__pycache__" -delete

# Create necessary directories with proper permissions
RUN mkdir -p /app/uploads /app/models /app/logs /app/tmp && \
    chown -R toyresale:toyresale /app && \
    chmod 755 /app && \
    chmod 700 /app/uploads /app/logs /app/tmp && \
    chmod 755 /app/models

# Copy application code with proper ownership
COPY --chown=toyresale:toyresale . .

# Security: Remove any potential sensitive files
RUN find /app -name "*.pyc" -delete && \
    find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true && \
    rm -f /app/.env* /app/secrets* 2>/dev/null || true

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Security: Switch to non-root user
USER toyresale

# Expose port
EXPOSE 8000

# Health check with proper timeout
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/v1/health', timeout=5)" || exit 1

# Security: Run with restricted capabilities
# Use exec form to avoid shell injection
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]