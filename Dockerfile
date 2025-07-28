# Use an official, lightweight Python runtime as a parent image
FROM python:3.13-slim

# Set environment variables for a clean and efficient Python environment
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /app

# --- This is the key change ---
# Instead of assuming the context is correct, we copy the entire repository content first.
# The source directory '.' refers to the root of the cloned repository content.
COPY . /app

# Now that all files are inside /app, we can run our commands.

# --- Optimized Installation ---
# Combine all pip installs into a single RUN command to reduce image layers.
RUN pip install -r requirements.txt
# Expose the port that Uvicorn will run on.
EXPOSE 8000

# The command to run your application when the container starts.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]