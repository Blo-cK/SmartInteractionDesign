FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . /app
ENV PYTHONPATH="/app/architecture:${PYTHONPATH}"
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "-k", "gevent", "-w", "1", "architecture.output_layer_monitor:app"]
