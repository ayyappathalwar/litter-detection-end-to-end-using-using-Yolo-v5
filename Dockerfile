FROM python:3.8-slim-bullseye
WORKDIR /app
COPY . /app

RUN apt update -y && apt install awscli -y

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 unzip -y \
    && pip install -r requirements.txt \
    && pip install . \
    && ln -sf /app/WasteDetection /app/wasteDetection \
    && cp /app/best.pt /app/yolov5/best.pt

ENV PYTHONPATH=/app
EXPOSE 8081
CMD ["python3", "app.py"]
