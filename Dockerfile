FROM python:3.11-slim

WORKDIR /app

# System deps needed for numpy/torch/pandas builds on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV CONFIG_PATH=config/config.yaml
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
EXPOSE 8501

# Default: run the API. Override this at `docker run` time to run any other
# script instead (preprocessing, training, tests).
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]