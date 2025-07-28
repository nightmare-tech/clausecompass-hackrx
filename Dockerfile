# Use an official, lightweight Python runtime as a parent image
FROM python:3.13-slim

# Set environment variables for a clean and efficient Python environment
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# --- This is the key fix ---
# Install system-level build dependencies FIRST.
# 'build-essential' includes gcc and other C/C++ compilers.
# 'pkg-config' is often needed by other build systems.
# We also clean up the apt cache afterwards to keep the image slim.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential pkg-config && \
    rm -rf /var/lib/apt/lists/*
# --- End of fix ---

# Set the working directory inside the container
WORKDIR /app

# Copy your application requirements and code
COPY requirements.txt .
COPY . .

# --- Optimized Installation ---
# Now that build tools are present, pip can compile packages like NumPy.
# We also download the ML model during the build.
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Uvicorn will run on
EXPOSE 8000

# The command to run your application when the container starts
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7999"]