# 1. Use a lightweight Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /mlops_app

# 3. Copy the requirements file
COPY requirements.txt .

# 4. Install PyTorch with NVIDIA CUDA 12.1 support (using extra-index-url to prevent dependency bugs)
RUN pip install --no-cache-dir --default-timeout=1000 torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# 5. Install the rest of the Hugging Face ecosystem
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the FastAPI app and the 35MB LoRA model weights into the container
COPY ./app ./app
COPY ./model ./model

# 7. Expose the API port
EXPOSE 8000

# 8. Start the Uvicorn server (Defaults to real GPU mode)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]