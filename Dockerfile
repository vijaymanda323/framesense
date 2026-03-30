# Use Python 3.13 slim image
FROM python:3.13-slim

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies using the extra index URL for PyTorch CPU
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# The application uses the PORT environment variable
# Use python main.py to start the server as it handles the port logic
CMD ["python", "main.py"]
