FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    htop \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv (to system Python)
RUN uv export --format requirements-txt > requirements.txt && \
    uv pip install --system -r requirements.txt

# Copy application code
COPY . .

# Define the entrypoint
ENTRYPOINT ["python", "-u", "-m", "models.cfr.run"]