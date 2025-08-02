FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./src .

ENV OLLAMA_HOST=http://ollama:11434

CMD ["bash"]

# CMD ["python", "app.py"]