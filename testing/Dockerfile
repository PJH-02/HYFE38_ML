FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install --no-cache-dir numpy pandas

COPY . /app

CMD ["python", "run_full_backtest.py"]
