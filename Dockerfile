# Use stable Python (important for numpy, torch, opencv)
FROM python:3.11-slim

# Prevent interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (OpenCV + torch support)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Expose port (Render uses PORT env internally)
EXPOSE 10000

# Start FastAPI with Uvicorn (IMPORTANT)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]