# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-cache

# Copy application source code
COPY src/ ./src/

# Copy data folder (RAG database)
COPY src/data/ ./src/data/

# Create config.json with default values
RUN echo '{\
  "llm": {\
    "model": "gemini-2.5-flash",\
    "temperature": 0.1\
  },\
  "vector_store": {\
    "persist_directory": "./src/data",\
    "collection_name": "obsidian",\
    "embedding_model": "models/gemini-embedding-001"\
  },\
  "retriever": {\
    "search_type": "similarity",\
    "k": 5\
  },\
  "obsidian": {\
    "path": ""\
  },\
  "pdf": {\
    "path": "./src/data/harrypotter.pdf"\
  },\
  "safety": {\
    "enabled": true,\
    "strict_mode": false\
  }\
}' > config.json

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI application
CMD ["uv", "run", "python", "src/app_api_handler.py"]