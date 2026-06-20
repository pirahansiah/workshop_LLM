FROM python:3.11-slim

LABEL maintainer="Farshid Pirahansiah"
LABEL description="CV Metaverse Workshop — 3D Multi-Camera Calibration"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

CMD ["python", "-m", "pytest", "tests/", "-v"]
