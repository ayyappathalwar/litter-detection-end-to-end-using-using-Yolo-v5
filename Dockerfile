FROM python:3.10-slim-bullseye
WORKDIR /app
COPY . /app


RUN apt update -y && apt install awscli -y

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 unzip -y && pip install -r requirements.txt && pip install -e .
EXPOSE 8081
CMD ["python3", "app.py"]
