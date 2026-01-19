# Start from a base image that has Python and PyTorch pre-installed
# This saves you from installing CUDA drivers manually
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
# Install system dependencies (often needed for Audio/MIDI)
# e.g., git, fluidsynth for midi, ffmpeg for audio
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    fluidsynth \
    && rm -rf /var/lib/apt/lists/*

# Copy your requirements file first (for caching)
COPY requirements.txt .

# Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Default command to run (optional)
CMD ["python", "main.py"]