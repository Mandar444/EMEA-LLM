FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (needed for compiling certain packages if necessary)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
# We structure this so that /app contains backend/ directory, preserving module namespaces
COPY backend/ ./backend

# Expose FastAPI port
EXPOSE 8000

# Set environment variable to resolve import namespaces
ENV PYTHONPATH=/app

# Start Uvicorn server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
