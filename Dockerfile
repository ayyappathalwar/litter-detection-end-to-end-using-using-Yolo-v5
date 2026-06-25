FROM python:3.8-slim-bullseye
WORKDIR /app
COPY . /app

RUN apt update -y && apt install awscli git -y

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 unzip -y \
    && rm -rf /app/yolov5 && git clone https://github.com/ultralytics/yolov5.git /app/yolov5 \
    && pip install -r requirements.txt \
    && pip install . \
    && ln -sf /app/WasteDetection /app/wasteDetection \
    && cp /app/best.pt /app/yolov5/best.pt \
    && echo "import pathlib; pathlib.WindowsPath = pathlib.PosixPath" > /usr/local/lib/python3.8/sitecustomize.py

ENV PYTHONPATH=/app
EXPOSE 8081
CMD ["python3", "app.py"]
