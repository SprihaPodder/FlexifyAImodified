FROM python:3.11-slim

# system deps for common packages (adjust if you need other libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc git libpq-dev libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# copy app source
COPY . /app

# expose port (Render provides $PORT)
EXPOSE 8000

# use shell form so env var expansion works
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]