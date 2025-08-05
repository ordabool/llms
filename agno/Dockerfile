FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for psycopg
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt

ENV AGNO_TELEMETRY=false

COPY ./src .

CMD ["bash"]

# CMD ["python", "app.py"]