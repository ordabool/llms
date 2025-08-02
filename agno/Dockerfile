FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

ENV AGNO_TELEMETRY=false

COPY ./src .

CMD ["bash"]

# CMD ["python", "app.py"]