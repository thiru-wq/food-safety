# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for OpenCV/PIL
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies (using tensorflow-cpu to save space/RAM)
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir tensorflow-cpu==2.16.1

# Copy the rest of the application code
COPY . .

# Expose the port FastAPI runs on
EXPOSE 7860

# Command to run the application
# Hugging Face Spaces expects the app to run on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
