import sys
import os
import time

from wasteDetection.pipeline.training_pipeline import TrainingPipeline
from wasteDetection.utils.main_utils import decode_image, encode_image
from wasteDetection.constants.application import APP_HOST, APP_PORT
from wasteDetection.logger.aws_logger import get_logger, log_aws_context, run_command

from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin


logger = get_logger()

app = Flask(__name__)
CORS(app)


# ---------------------------------------------------------------------------
# Request / response logging middleware
# ---------------------------------------------------------------------------

@app.before_request
def _log_request():
    request._start_time = time.time()
    logger.info(
        "REQUEST  %s %s  client=%s",
        request.method,
        request.path,
        request.remote_addr,
    )


@app.after_request
def _log_response(response):
    elapsed_ms = (time.time() - getattr(request, "_start_time", time.time())) * 1000
    logger.info(
        "RESPONSE %s %s  status=%d  elapsed=%.1fms",
        request.method,
        request.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"


@app.route("/train")
def trainRoute():
    logger.info("Training pipeline triggered")
    try:
        obj = TrainingPipeline
        obj.run_pipeline
        logger.info("Training pipeline completed")
        return "Training Successfully"
    except Exception:
        logger.exception("Training pipeline failed")
        return Response("Training failed — check logs", status=500)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST", "GET"])
@cross_origin()
def predictRoute():
    try:
        image = request.json["image"]
        decode_image(image, clApp.filename)
        logger.info("Input image decoded, running YOLOv5 detection")

        exit_code = run_command(
            "cd yolov5/ && python detect.py --weights best.pt --img 416 --conf 0.5 --source ../data/inputImage.jpg",
            logger,
        )
        if exit_code != 0:
            logger.error("YOLOv5 detect.py exited with code %d", exit_code)
            return Response("Detection failed — check logs", status=500)

        opencodedbase64 = encode_image("yolov5/runs/detect/exp/inputImage.jpg")
        result = {"image": opencodedbase64.decode("utf-8")}

        run_command("rm -rf yolov5/runs", logger)
        logger.info("Prediction complete")

    except ValueError as val:
        logger.error("ValueError in predictRoute: %s", val)
        return Response("Value not found inside json data")
    except KeyError as ke:
        logger.error("KeyError in predictRoute: missing key %s", ke)
        return Response("Key value error — incorrect key passed")
    except Exception:
        logger.exception("Unexpected error in predictRoute")
        result = "Invalid Input"

    return jsonify(result)


@app.route("/live", methods=["GET"])
@cross_origin
def predictive():
    try:
        logger.info("Live camera detection starting")
        run_command(
            "cd yolov5/ && python detect.py --weight best.pt --img 416 --conf 0.5 --source 0",
            logger,
        )
        run_command("rm -rf yolov5/runs", logger)
        return "Camera starting!!"

    except ValueError as val:
        logger.error("ValueError in predictive: %s", val)
        return Response("Value not found inside json data")
    except Exception:
        logger.exception("Unexpected error in predictive")
        return Response("Live detection failed — check logs", status=500)


if __name__ == "__main__":
    clApp = ClientApp()
    log_aws_context(logger)
    logger.info("Starting Flask on %s:%s", APP_HOST, APP_PORT)
    app.run(host=APP_HOST, port=APP_PORT)
