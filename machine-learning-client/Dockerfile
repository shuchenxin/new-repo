FROM python:3.11-slim

WORKDIR /app

# System deps for audio processing (SoundFile/libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
CMD ["python", "-m", "app.worker"]
