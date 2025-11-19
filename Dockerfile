# Use Python 3.10 Slim (Lightweight)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- THE FIX: Install PyTorch CPU Version Explicitly FIRST ---
# This forces the small CPU version (approx 200MB) instead of the 5GB GPU version
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install the rest
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Keep container running
CMD ["tail", "-f", "/dev/null"]