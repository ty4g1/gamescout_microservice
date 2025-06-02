# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12.10
FROM python:${PYTHON_VERSION}-slim as base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user with a proper home directory
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/bin/bash" \
    --uid "${UID}" \
    appuser

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Set cache directories that appuser can access
ENV HF_HOME=/app/cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/cache

# Create cache directory and set permissions
RUN mkdir -p /app/cache && chown -R appuser:appuser /app/cache

# Switch to appuser BEFORE downloading the model
USER appuser

# Now download the model as appuser
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the source code into the container
COPY --chown=appuser:appuser . .

# Expose the port that the application listens on
EXPOSE 8000

# Run the application
CMD ["fastapi", "run", "app/main.py"]