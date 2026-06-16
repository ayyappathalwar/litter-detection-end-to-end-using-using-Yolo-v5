import sys, os
from wasteDetection.pipeline.training_pipeline import TrainingPipeline  
from wasteDetection.utils.main_utils import decode_image, encode_image
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin
from wasteDetection.constants.application import APP_HOST, APP_PORT


app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"

    
    
@app.route("/train")
def trainRoute():
    obj = TrainingPipeline
    obj.run_pipeline
    return "Training Successfully"
    

@app.route("/")
def home():
    return render_template("index.html")

    
@app.route("/predict", methods=['POST','GET'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decode_image(image, clApp.filename)

        os.system("cd yolov5 && python detect.py --weights best.pt --img 640 --conf 0.3 --source ../data/inputImage.jpg")

        opencodedbase64 = encode_image("yolov5/runs/detect/exp/inputImage.jpg")
        result = {"image": opencodedbase64.decode('utf-8')}
        os.system("rmdir /s /q yolov5\\runs")

    except ValueError as val:
        print(val)
        return Response("Value not found inside json data")
    except KeyError:
        return Response("Key value error incorret key passed")
    except Exception as e:
        print(e)
        result = "Invalid Input"

    return jsonify(result)
    

@app.route("/live", methods=['GET'])
@cross_origin
def predictive():
    try:
        os.system("cd yolov5 && python detect.py --weights best.pt --img 416 --conf 0.5 --source 0")
        os.system("rmdir /s /q yolov5\\runs")
        return "Camera starting!!"
        
    except ValueError as val:
        print(val)
        return Response("Value not found inside json data")


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host=APP_HOST, port=APP_PORT)
