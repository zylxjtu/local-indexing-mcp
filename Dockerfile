FROM python:3.11-slim

# Accept port as build argument with default value
ARG FASTMCP_PORT=8978

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY code_indexer_server.py .

# Create directory for ChromaDB
RUN mkdir -p /app/chroma_db

# Expose the FastMCP port (uses ARG from build-time)
EXPOSE ${FASTMCP_PORT}

# Command to run the application
CMD ["python", "code_indexer_server.py"] 