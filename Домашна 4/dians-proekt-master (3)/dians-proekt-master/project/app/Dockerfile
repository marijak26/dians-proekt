FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt
RUN apt-get update && apt-get install -y tesseract-ocr


COPY .. /app

EXPOSE 5000

CMD ["python", "app.py"]
