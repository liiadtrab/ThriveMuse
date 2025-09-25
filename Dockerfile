# Use PyTorch base image with CUDA support (closest to user's setup)
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade to exact PyTorch version matching user's setup
RUN pip uninstall torch torchvision torchaudio -y
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install exact dependencies from user's working environment
COPY requirements_exact.txt /tmp/requirements_exact.txt
RUN pip install -r /tmp/requirements_exact.txt

# Clone MuseTalk
RUN git clone https://github.com/TMElyralab/MuseTalk.git /app/MuseTalk
WORKDIR /app/MuseTalk

# Download models (this will take a while)
RUN python scripts/download_weights.py

# Copy our wrapper, API server and assets
COPY musetalk_wrapper.py /app/
COPY api_server.py /app/
COPY assets/avatar_video.mp4 /app/assets/

# Create temp directory (Linux path for container)
RUN mkdir -p /tmp/results

# Update wrapper paths for Linux environment
ENV MUSETALK_PATH=/app/MuseTalk
ENV FFMPEG_BIN=/usr/bin
ENV TEMP_DIR=/tmp/results

# Expose port for API
EXPOSE 8080

# Set Python path
ENV PYTHONPATH=/app:/app/MuseTalk

# Run our API server
CMD ["python", "/app/api_server.py"]
