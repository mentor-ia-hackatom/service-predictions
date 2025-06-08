FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including libgomp
RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 4001

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4001"]