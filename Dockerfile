FROM python:3.11-slim

# Install necessary tools
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Disable Poetry virtual environments
RUN poetry config virtualenvs.create false

# Set Hugging Face mirror
ENV HF_ENDPOINT=https://hf-mirror.com

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* Makefile ./

# Install dependencies
RUN poetry install --no-root --with dev --with test

# Copy source code
COPY . .

# Set PYTHONPATH
ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 8000

# Start API service
CMD ["poetry", "run", "uvicorn", "memos.api.product_api:app", "--host", "0.0.0.0", "--port", "8000"]
