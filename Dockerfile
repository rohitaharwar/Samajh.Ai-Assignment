FROM python:3.9-slim

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install opencv-python deep_sort_realtime pandas requests

CMD ["python", "main.py"]