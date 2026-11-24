# Use a lightweight Python base image
FROM python:3.10-slim

# Print Python logs immediately in Docker and cloud consoles
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /mlops_app

# Copy requirements first so Docker caches installed packages between code edits
COPY requirements.txt .

# Install PyTorch (Linux PyPI wheels already include NVIDIA CUDA support)
RUN pip install --no-cache-dir --default-timeout=1000 torch

# Install FastAPI and the Hugging Face / QLoRA libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app and your fine-tuned v3 LoRA adapter into the container
COPY ./app ./app
COPY ./model ./model

# Expose the API port
EXPOSE 8000

# Start the FastAPI server on all network interfaces
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]