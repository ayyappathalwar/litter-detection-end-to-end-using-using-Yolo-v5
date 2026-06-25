# MLOps from Scratch — Complete Learning Guide
### Building a Waste Detection System using YOLOv5, Flask, Docker, and AWS

> **Who this is for:** Someone who knows basic Python and wants to understand
> how a real-world ML project is structured, deployed, and maintained on the cloud.
> Every concept is explained from first principles.

---

## Table of Contents

1. [What is MLOps? (The Big Picture)](#1-what-is-mlops)
2. [What Are We Building?](#2-what-are-we-building)
3. [Phase 1 — Project Structure & Setup](#3-phase-1--project-structure--setup)
4. [Phase 2 — Configuration & Constants](#4-phase-2--configuration--constants)
5. [Phase 3 — Logging & Exception Handling](#5-phase-3--logging--exception-handling)
6. [Phase 4 — Data Ingestion](#6-phase-4--data-ingestion)
7. [Phase 5 — Data Validation](#7-phase-5--data-validation)
8. [Phase 6 — Model Training (YOLOv5)](#8-phase-6--model-training-yolov5)
9. [Phase 7 — Flask Web Application](#9-phase-7--flask-web-application)
10. [Phase 8 — Docker (Containerisation)](#10-phase-8--docker)
11. [Phase 9 — AWS Infrastructure](#11-phase-9--aws-infrastructure)
12. [Phase 10 — CI/CD Pipeline (GitHub Actions)](#12-phase-10--cicd-pipeline)
13. [Phase 11 — Logging for AWS Deployment](#13-phase-11--logging-for-aws-deployment)
14. [How Everything Connects — End-to-End Flow](#14-how-everything-connects)
15. [Common Mistakes & How to Avoid Them](#15-common-mistakes)

---

## 1. What is MLOps?

### Think of it like a restaurant

A **data scientist** is like a chef who experiments in the kitchen to create a great recipe (a model).

But a great recipe alone doesn't feed customers. You need:
- A kitchen (infrastructure) to cook at scale
- A system that orders the right ingredients (data pipeline)
- Quality checks on ingredients (data validation)
- A reliable way to serve the food (deployment)
- Cameras in the kitchen so you know when something goes wrong (logging & monitoring)
- A way to improve the recipe without closing the restaurant (CI/CD)

**MLOps** = Machine Learning + Operations.
It is the practice of taking an ML model from a notebook into a production system that:
- runs reliably
- can be updated without breaking
- logs everything that happens
- is tested before being pushed live

### The MLOps Lifecycle

```
[Data] → [Ingest] → [Validate] → [Train] → [Package] → [Deploy] → [Monitor]
                                                              ↑            |
                                                              |____________|
                                                          (retrain if model drifts)
```

Every arrow in the diagram above is a stage you will write code for.

---

## 2. What Are We Building?

A system that:
1. **Accepts an image** via a web API
2. **Runs YOLOv5** object detection to find litter/waste in the image
3. **Returns the annotated image** back to the caller
4. **Trains the model** on new data on demand
5. **Runs in a Docker container** so it works identically on any machine
6. **Deploys to AWS EC2** automatically whenever you push code to GitHub

### System Architecture

```
USER (browser / app)
       |
       | HTTP POST /predict  (base64 image)
       ↓
┌─────────────────────────────────────┐
│         Flask Web Server            │  ← runs inside Docker on EC2
│  /           → home page            │
│  /train      → kick off training    │
│  /predict    → run detection        │
│  /live       → live camera feed     │
└─────────────────────────────────────┘
       |
       | calls
       ↓
┌─────────────────────────────────────┐
│     YOLOv5 detect.py                │  ← the actual ML model
│  Input:  inputImage.jpg             │
│  Output: runs/detect/exp/*.jpg      │
└─────────────────────────────────────┘

Deployment pipeline:
GitHub Push → GitHub Actions CI → Docker Build → Push to ECR → Pull on EC2 → Run
```

---

## 3. Phase 1 — Project Structure & Setup

### Why project structure matters

When a project is small (one notebook), structure does not matter.
When a project grows to 10+ files used by a team, bad structure = chaos.

A good structure means:
- You always know where to find a file
- Other people can onboard quickly
- Automated tools (tests, CI) know where to look

### The folder structure we will build

```
waste-detection/
│
├── wasteDetection/              ← main Python package (our application code)
│   ├── __init__.py
│   ├── components/              ← individual pipeline steps
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   └── model_trainer.py
│   ├── constants/               ← all magic numbers & strings live here
│   │   ├── __init__.py
│   │   ├── application.py
│   │   └── training_pipeline/
│   │       └── __init__.py
│   ├── entitys/                 ← data classes that describe config & results
│   │   ├── config_entity.py
│   │   └── artifact_entity.py
│   ├── exceptions/              ← custom error classes
│   │   └── __init__.py
│   ├── logger/                  ← logging setup
│   │   ├── __init__.py
│   │   └── aws_logger.py
│   ├── pipeline/                ← orchestrates components in order
│   │   └── training_pipeline.py
│   └── utils/                   ← shared helper functions
│       └── main_utils.py
│
├── data/                        ← raw input data (gitignored if large)
├── artifacts/                   ← pipeline outputs (models, reports)
├── log/                         ← local log files
├── templates/                   ← HTML files for the web UI
├── yolov5/                      ← YOLOv5 source code (cloned separately)
│
├── app.py                       ← Flask entry point
├── setup.py                     ← makes wasteDetection installable as a package
├── requirements.txt             ← Python dependencies
├── Dockerfile                   ← container recipe
├── .github/
│   └── workflows/
│       └── main.yaml            ← CI/CD pipeline definition
└── README.md
```

### Step 1 — Create the package with setup.py

**What is a Python package?**
Normally when you write `import something`, Python looks in a fixed list of places.
A `setup.py` tells Python "this folder is also a package you can import from".

Without it, you would have to mess with `sys.path` in every file — messy and fragile.

```python
# setup.py
from setuptools import find_packages, setup

setup(
    name="wasteDetection",          # name of our package
    version="0.0.1",
    author="Your Name",
    packages=find_packages(),       # automatically finds every folder with __init__.py
    install_requires=[],            # list dependencies here OR keep them in requirements.txt
)
```

**Install it in editable mode** (changes to source are reflected immediately):
```bash
pip install -e .
```

After this, `from wasteDetection.logger import ...` works from anywhere.

### Step 2 — requirements.txt

This file lists every library your project depends on, with exact versions.
This is critical for reproducibility — the same file works on your laptop,
on a teammate's laptop, and inside Docker.

```
flask
flask-cors
from-root          # helps find the project root directory reliably
PyYAML             # for reading .yaml config files
opencv-python-headless  # image processing (headless = no GUI, needed in Docker)
torch              # PyTorch (required by YOLOv5)
```

**Install everything:**
```bash
pip install -r requirements.txt
```

### Step 3 — The __init__.py files

Every folder that is part of your Python package needs an `__init__.py` file.
It can be completely empty — its mere existence tells Python "this folder is a module".

```python
# wasteDetection/__init__.py
# (empty file — just needs to exist)
```

---

## 4. Phase 2 — Configuration & Constants

### The problem this solves

Imagine you hard-code the image size as `416` in three different files.
Then your team decides to change it to `640`.
Now you have to hunt through every file to change it.
One missed change = a bug.

**Solution:** Put every configurable value in one place (constants).

### Step 1 — Application constants

```python
# wasteDetection/constants/application.py

APP_HOST = "0.0.0.0"    # listen on all network interfaces (required for Docker)
APP_PORT = 8081          # the port Flask runs on
```

**Why `0.0.0.0` and not `localhost`?**
`localhost` means "only accept connections from this same machine."
Inside Docker, the Flask server and your browser are on different "machines."
`0.0.0.0` means "accept connections from anywhere" — this is what makes the app
accessible from outside the container.

### Step 2 — Training pipeline constants

```python
# wasteDetection/constants/training_pipeline/__init__.py

# Where to store raw downloaded data
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

# Where to find the data source
DATA_DOWNLOAD_URL: str = "https://your-s3-bucket/dataset.zip"

# Model training settings
BATCH_SIZE: int = 16
IMAGE_SIZE: int = 416
NUM_EPOCHS: int = 50
MODEL_ARCHITECTURE: str = "yolov5s.pt"   # s=small, m=medium, l=large, x=extra-large
```

### Step 3 — Entity classes (Config & Artifact)

**What is an entity?**
An entity is a simple data container — like a named tuple or a struct in C.
Instead of passing 10 separate variables into a function, you group them
into an entity and pass one object.

**Config entity** = "what do I need to *start* a task?"

```python
# wasteDetection/entitys/config_entity.py
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    # The root directory where all pipeline outputs go
    artifacts_dir: str

    # Sub-directory for data ingestion outputs
    data_ingestion_dir: str

    # Path where downloaded raw data is stored
    feature_store_file_path: str

    # URL to download the dataset from
    data_download_url: str
```

> `@dataclass` is a Python decorator that automatically generates
> `__init__`, `__repr__`, and `__eq__` for a class based on its
> field annotations. It saves you from writing boilerplate.

**Artifact entity** = "what did a task *produce*?"

```python
# wasteDetection/entitys/artifact_entity.py
from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    # Path to the data after ingestion is complete
    data_zip_file_path: str
    feature_store_path: str
```

**Why separate config from artifact?**
- Config is *input* — it tells a component what to do and where to put things.
- Artifact is *output* — it tells the next component where to find what was produced.
- This pattern makes pipelines composable: output of step N becomes input of step N+1.

---

## 5. Phase 3 — Logging & Exception Handling

### Why logging is more important than print()

`print()` works fine locally. In production (especially on AWS), you need:
- **Timestamps** — when did this happen?
- **Log levels** — is this just INFO, or is it a WARNING or ERROR?
- **Persistence** — logs written to a file survive after the program crashes
- **CloudWatch integration** — AWS can collect logs from Docker containers automatically

### The difference between log levels

| Level    | When to use                                        |
|----------|----------------------------------------------------|
| DEBUG    | Detailed info for diagnosing problems (verbose)    |
| INFO     | Normal operation — "request received", "job done"  |
| WARNING  | Something unexpected but not broken                |
| ERROR    | Something went wrong, the operation failed         |
| CRITICAL | The whole system might go down                     |

In production you usually show INFO and above.
In development you show DEBUG and above.

### Step 1 — Basic logger (local)

```python
# wasteDetection/logger/__init__.py
import logging
import os
from datetime import datetime
from from_root import from_root   # finds the project root reliably

# Create a timestamped filename like "2025-01-15_10-30-00.log"
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# Put the log file inside <project_root>/log/
log_dir = os.path.join(from_root(), "log")
os.makedirs(log_dir, exist_ok=True)   # create the folder if it doesn't exist
LOG_FILE_PATH = os.path.join(log_dir, LOG_FILE)

# Configure the root logger
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
```

**How to use it in any other file:**
```python
import logging
logger = logging.getLogger(__name__)   # __name__ gives the current module name

logger.info("Starting data ingestion")
logger.error("Failed to download data: %s", error_message)
```

### Step 2 — Custom Exception class

Python has built-in exceptions like `FileNotFoundError` and `ValueError`.
But they don't tell you *which file* and *which line number* the error occurred in.
We write a custom exception that includes all of that.

```python
# wasteDetection/exceptions/__init__.py
import sys

def error_message_detail(error, error_detail: sys) -> str:
    """
    Extracts the filename and line number where the exception occurred.

    error_detail is the sys module — we pass sys so we can call sys.exc_info()
    which returns (type, value, traceback). We only need the traceback.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    return (
        f"Error in script: [{file_name}] "
        f"at line [{line_number}] "
        f"with message [{str(error)}]"
    )


class AppException(Exception):
    """
    Custom exception that includes file name, line number, and error message.

    Usage:
        try:
            ...
        except Exception as e:
            raise AppException(e, sys)
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
```

**Example of using it:**
```python
import sys
from wasteDetection.exceptions import AppException

try:
    result = 10 / 0
except Exception as e:
    raise AppException(e, sys)
# Output: Error in script: [my_file.py] at line [4] with message [division by zero]
```

---

## 6. Phase 4 — Data Ingestion

Data ingestion = "getting the data from wherever it lives into a form your pipeline can use."

In our case: download a .zip file from a URL, unzip it, and store it in `artifacts/`.

### Step 1 — Configuration for this component

```python
# In wasteDetection/entitys/config_entity.py

@dataclass
class DataIngestionConfig:
    artifacts_dir: str = "artifacts"
    data_ingestion_dir: str = "artifacts/data_ingestion"
    feature_store_file_path: str = "artifacts/data_ingestion/feature_store"
    data_download_url: str = "https://..."  # your data source URL
```

### Step 2 — The component itself

```python
# wasteDetection/components/data_ingestion.py
import os
import sys
import zipfile
import logging
import urllib.request

from wasteDetection.entitys.config_entity import DataIngestionConfig
from wasteDetection.entitys.artifact_entity import DataIngestionArtifact
from wasteDetection.exceptions import AppException

logger = logging.getLogger(__name__)


class DataIngestion:
    """
    Responsible for downloading and extracting the dataset.

    Think of this as the "receiving dock" of our pipeline.
    Raw data comes in, gets unpacked, and stored in a known location.
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        # Store the config — we'll use it in every method
        self.data_ingestion_config = data_ingestion_config

    def download_data(self) -> str:
        """
        Downloads the dataset zip file from the configured URL.
        Returns the path where the zip file was saved.
        """
        try:
            url = self.data_ingestion_config.data_download_url
            zip_dir = self.data_ingestion_config.data_ingestion_dir
            os.makedirs(zip_dir, exist_ok=True)   # create folder if not present

            zip_path = os.path.join(zip_dir, "data.zip")

            logger.info("Downloading data from: %s", url)
            urllib.request.urlretrieve(url, zip_path)
            logger.info("Download complete. Saved to: %s", zip_path)

            return zip_path

        except Exception as e:
            raise AppException(e, sys)

    def extract_zip_file(self, zip_path: str) -> str:
        """
        Unzips the downloaded file into the feature store directory.
        Returns the path to the extracted folder.
        """
        try:
            feature_store_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(feature_store_path, exist_ok=True)

            logger.info("Extracting %s to %s", zip_path, feature_store_path)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(feature_store_path)

            logger.info("Extraction complete")
            return feature_store_path

        except Exception as e:
            raise AppException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Orchestrates the full ingestion: download → extract → return artifact.

        This is the "public API" of the component.
        The pipeline only calls this method, not the individual methods above.
        """
        logger.info("Starting data ingestion")
        try:
            zip_path = self.download_data()
            feature_store_path = self.extract_zip_file(zip_path)

            # Package up what we produced into an artifact
            artifact = DataIngestionArtifact(
                data_zip_file_path=zip_path,
                feature_store_path=feature_store_path,
            )

            logger.info("Data ingestion complete: %s", artifact)
            return artifact

        except Exception as e:
            raise AppException(e, sys)
```

**Key design patterns here:**
- Each component takes a *config entity* in `__init__` — this makes it testable.
  You can create a config pointing to a test directory without changing the class.
- Each component returns an *artifact entity* — this is the handshake between steps.
- All exceptions are caught and re-raised as `AppException` — consistent error format.

---

## 7. Phase 5 — Data Validation

After downloading data, you need to verify it is what you expected.
Imagine downloading a corrupted zip or a dataset missing its labels folder.
Without validation, the training step will crash with a confusing error 3 hours later.
With validation, it crashes immediately with a clear message.

### What to validate

For a YOLOv5 dataset you typically check:
- The expected folders exist (`images/`, `labels/`)
- YAML config file is present
- There is at least one image
- Labels folder is not empty

```python
# wasteDetection/components/data_validation.py
import os
import sys
import logging
import shutil

from wasteDetection.entitys.config_entity import DataValidationConfig
from wasteDetection.entitys.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from wasteDetection.exceptions import AppException

logger = logging.getLogger(__name__)


class DataValidation:
    """
    Checks that the ingested data has the expected structure.

    A YOLOv5 dataset must have:
      - data.yaml          (describes classes and paths)
      - train/images/      (training images)
      - train/labels/      (bounding box annotations)
      - valid/images/
      - valid/labels/
    """

    REQUIRED_DIRS = ["train/images", "train/labels", "valid/images", "valid/labels"]

    def __init__(
        self,
        data_validation_config: DataValidationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
    ):
        self.config = data_validation_config
        self.ingestion_artifact = data_ingestion_artifact

    def validate_all_files_exist(self) -> bool:
        """Returns True only if all required directories exist and are non-empty."""
        try:
            data_dir = self.ingestion_artifact.feature_store_path
            missing = []

            for required_dir in self.REQUIRED_DIRS:
                full_path = os.path.join(data_dir, required_dir)
                if not os.path.isdir(full_path):
                    missing.append(required_dir)
                elif len(os.listdir(full_path)) == 0:
                    missing.append(f"{required_dir} (empty)")

            if missing:
                logger.error("Validation failed. Missing: %s", missing)
                return False

            logger.info("All required directories found and non-empty")
            return True

        except Exception as e:
            raise AppException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """Run validation and return an artifact with the result."""
        logger.info("Starting data validation")
        try:
            is_valid = self.validate_all_files_exist()

            artifact = DataValidationArtifact(
                validation_status=is_valid,
                valid_data_dir=self.ingestion_artifact.feature_store_path if is_valid else None,
                invalid_data_dir=None if is_valid else self.ingestion_artifact.feature_store_path,
            )

            if not is_valid:
                raise AppException("Data validation failed — dataset structure is incorrect", sys)

            logger.info("Data validation passed")
            return artifact

        except Exception as e:
            raise AppException(e, sys)
```

---

## 8. Phase 6 — Model Training (YOLOv5)

YOLOv5 is a ready-made object detection framework.
We do not write the model architecture from scratch — we use it as a tool.

Our job is to:
1. Configure training parameters
2. Call YOLOv5's training script with the right arguments
3. Copy the best trained weights to a known location for the app to use

```python
# wasteDetection/components/model_trainer.py
import os
import sys
import logging
import subprocess
import shutil

from wasteDetection.entitys.config_entity import ModelTrainerConfig
from wasteDetection.entitys.artifact_entity import ModelTrainerArtifact
from wasteDetection.exceptions import AppException

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains the YOLOv5 model on our waste detection dataset.

    We call YOLOv5's train.py script as a subprocess.
    This means YOLOv5 handles the heavy lifting (architecture, loss functions,
    data augmentation) and we just configure it.
    """

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_validation_artifact,
    ):
        self.config = model_trainer_config
        self.data_dir = data_validation_artifact.valid_data_dir

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logger.info("Starting model training")
        try:
            # Build the command to call YOLOv5's training script
            train_cmd = (
                f"cd yolov5 && python train.py "
                f"--img {self.config.image_size} "
                f"--batch {self.config.batch_size} "
                f"--epochs {self.config.num_epochs} "
                f"--data {self.data_dir}/data.yaml "
                f"--weights {self.config.weight_name} "
                f"--name waste_detection"
            )

            logger.info("Running: %s", train_cmd)

            # subprocess.run captures output; os.system discards it
            result = subprocess.run(
                train_cmd,
                shell=True,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.error("Training failed:\n%s", result.stderr)
                raise AppException("YOLOv5 training failed", sys)

            logger.info("Training output:\n%s", result.stdout[-2000:])   # last 2000 chars

            # YOLOv5 saves the best model to yolov5/runs/train/waste_detection/weights/best.pt
            trained_model = "yolov5/runs/train/waste_detection/weights/best.pt"

            # Copy it to a stable location so the app always knows where to find it
            os.makedirs(self.config.model_trainer_dir, exist_ok=True)
            dest = os.path.join(self.config.model_trainer_dir, "best.pt")
            shutil.copy(trained_model, dest)
            logger.info("Model saved to: %s", dest)

            # Also copy to yolov5/ so detect.py can find it easily
            shutil.copy(dest, "yolov5/best.pt")

            return ModelTrainerArtifact(trained_model_path=dest)

        except Exception as e:
            raise AppException(e, sys)
```

### Key concept: subprocess vs os.system

```python
# BAD — you can't see what happened
os.system("python train.py")

# GOOD — you capture and log all output
result = subprocess.run("python train.py", shell=True, capture_output=True, text=True)
print(result.stdout)    # training logs
print(result.stderr)    # errors
print(result.returncode)  # 0 = success, anything else = failure
```

---

## 9. Phase 7 — Flask Web Application

Flask is a lightweight Python web framework.
It listens for HTTP requests and runs your Python code in response.

### HTTP basics (what you need to know)

A **request** has:
- A **method**: GET (fetch data) or POST (send data)
- A **path**: `/predict`, `/train`, etc.
- A **body**: the data you're sending (e.g., a base64-encoded image)

A **response** has:
- A **status code**: 200 = OK, 404 = not found, 500 = server error
- A **body**: the data you're sending back (e.g., an annotated image)

### Step 1 — Basic Flask app structure

```python
# app.py
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin

app = Flask(__name__)   # create the app
CORS(app)               # allow requests from other origins (browser security)
```

**What is CORS?**
Browsers block JavaScript from calling APIs on a different domain by default.
CORS (Cross-Origin Resource Sharing) is a mechanism to allow it.
`flask_cors` adds the right HTTP headers automatically so browsers don't block your API.

### Step 2 — Routes (endpoints)

A route = a URL pattern + a Python function.
When Flask receives a request matching the URL, it calls the function.

```python
@app.route("/")                         # matches the URL "/"
def home():
    return render_template("index.html")  # return HTML page

@app.route("/predict", methods=["POST", "GET"])  # matches /predict
@cross_origin()                                  # allow CORS for this specific route
def predictRoute():
    # request.json is the JSON body sent by the caller
    image_data = request.json["image"]
    ...
    return jsonify({"image": result_base64})     # return JSON
```

### Step 3 — The full app.py with explanations

```python
# app.py
import sys
import os
import time
import logging

from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin

from wasteDetection.utils.main_utils import decode_image, encode_image
from wasteDetection.constants.application import APP_HOST, APP_PORT
from wasteDetection.logger.aws_logger import get_logger, log_aws_context, run_command

# Get our logger (outputs to stdout AND a file)
logger = get_logger()

app = Flask(__name__)
CORS(app)


# ── Middleware ─────────────────────────────────────────────────────────────
# before_request runs before EVERY route handler
# after_request runs after EVERY route handler
# This is the cleanest way to log all requests — no repetition

@app.before_request
def log_incoming():
    request._start_time = time.time()       # save start time on the request object
    logger.info("→ %s %s from %s", request.method, request.path, request.remote_addr)

@app.after_request
def log_outgoing(response):
    elapsed = (time.time() - request._start_time) * 1000
    logger.info("← %s %s  status=%d  %.1fms", request.method, request.path,
                response.status_code, elapsed)
    return response                         # MUST return response


# ── Helper class ───────────────────────────────────────────────────────────
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"    # where we save the incoming image


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/train")
def trainRoute():
    """Kick off a full training pipeline run."""
    try:
        from wasteDetection.pipeline.training_pipeline import TrainingPipeline
        logger.info("Training triggered via /train endpoint")
        TrainingPipeline().run_pipeline()
        return "Training Successful"
    except Exception:
        logger.exception("Training failed")
        return Response("Training failed — check logs", status=500)


@app.route("/predict", methods=["POST", "GET"])
@cross_origin()
def predictRoute():
    """
    Accept a base64-encoded image, run YOLOv5 detection, return the result.

    Request body (JSON):
        { "image": "<base64 string>" }

    Response body (JSON):
        { "image": "<base64 string of annotated image>" }
    """
    try:
        image_b64 = request.json["image"]
        decode_image(image_b64, clApp.filename)    # write image to disk
        logger.info("Image received and decoded")

        # Run detection — output is logged, not silently discarded
        exit_code = run_command(
            "cd yolov5/ && python detect.py "
            "--weights best.pt --img 416 --conf 0.5 "
            "--source ../data/inputImage.jpg",
            logger,
        )

        if exit_code != 0:
            return Response("Detection failed — check server logs", status=500)

        # Read the output image and encode it back to base64
        result_b64 = encode_image("yolov5/runs/detect/exp/inputImage.jpg")
        result = {"image": result_b64.decode("utf-8")}

        run_command("rm -rf yolov5/runs", logger)   # clean up temp files
        return jsonify(result)

    except ValueError as e:
        logger.error("ValueError: %s", e)
        return Response("Value not found in request body")
    except KeyError as e:
        logger.error("Missing key: %s", e)
        return Response("Missing key in request JSON")
    except Exception:
        logger.exception("Unexpected error in /predict")
        return Response("Unexpected error", status=500)


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    clApp = ClientApp()
    log_aws_context(logger)         # print AWS context info to logs at startup
    logger.info("Server starting on %s:%s", APP_HOST, APP_PORT)
    app.run(host=APP_HOST, port=APP_PORT)
```

### Key concept: Why `host="0.0.0.0"`?

```
Your laptop:
  Flask on 0.0.0.0:8081 → accepts connections from:
    - your browser (localhost:8081) ✓
    - another person on your network (192.168.x.x:8081) ✓

Docker container:
  Flask on 0.0.0.0:8081 → accepts connections from:
    - outside the container (your EC2 public IP:8081) ✓

If you used 127.0.0.1 (localhost) inside Docker:
    - outside the container ✗ (connection refused)
```

---

## 10. Phase 8 — Docker

### What is Docker and why does it exist?

**The problem:** "It works on my machine."

A program depends on:
- The Python version
- Every installed library (and their versions)
- Environment variables
- Operating system libraries

When any of these differ between machines, the program breaks.

**The solution:** Docker packages your application AND its entire environment
into a single file called an **image**.
When you run the image, it creates a **container** — an isolated mini-computer
that is identical everywhere it runs.

### Key vocabulary

| Term       | Analogy                                          |
|------------|--------------------------------------------------|
| Image      | A recipe + all ingredients, frozen in time       |
| Container  | A dish cooked from that recipe, currently running|
| Dockerfile | The instructions for building the image          |
| Registry   | A place to store and share images (like GitHub)  |
| ECR        | AWS's Docker registry                            |

### Step 1 — Write the Dockerfile

```dockerfile
# Dockerfile

# ── Base image ─────────────────────────────────────────────────────────────
# We start FROM an official Python 3.8 image.
# "slim-bullseye" means Debian Linux, minimal packages (small image size).
FROM python:3.8-slim-bullseye

# ── Set working directory ─────────────────────────────────────────────────
# All subsequent commands run inside /app.
# This is where our code will live inside the container.
WORKDIR /app

# ── Copy source code ──────────────────────────────────────────────────────
# Copy everything from the current directory on your machine into /app.
# Do this BEFORE pip install so Docker cache works correctly.
COPY . /app

# ── Install system dependencies ───────────────────────────────────────────
# awscli: needed to interact with AWS (download models, etc.)
# ffmpeg, libsm6, libxext6: needed by OpenCV for image processing
# unzip: needed to extract data archives
RUN apt-get update -y && \
    apt-get install -y awscli ffmpeg libsm6 libxext6 unzip && \
    rm -rf /var/lib/apt/lists/*   # clean up to keep image size small

# ── Install Python dependencies ───────────────────────────────────────────
RUN pip install --no-cache-dir -r requirements.txt

# ── Install our package ───────────────────────────────────────────────────
RUN pip install --no-cache-dir -e .

# ── Expose the port ───────────────────────────────────────────────────────
# This documents which port the container uses.
# You still need to map it with -p 8081:8081 when running.
EXPOSE 8081

# ── Entry point ───────────────────────────────────────────────────────────
# This command runs when the container starts.
CMD ["python3", "app.py"]
```

### Step 2 — Build and run locally

```bash
# Build the image and tag it as "waste-detection:latest"
docker build -t waste-detection:latest .

# Run a container from the image
# -d: run in background (detached)
# -p 8081:8081: map port 8081 on your machine to port 8081 in the container
# --name waste: give the container a name so you can reference it
docker run -d -p 8081:8081 --name waste waste-detection:latest

# Check it's running
docker ps

# See its logs
docker logs -f waste          # -f means "follow" (live tail)

# Stop and remove it
docker stop waste && docker rm waste
```

### Docker layer caching — why order matters

Docker builds images in **layers**. Each instruction in the Dockerfile is a layer.
If a layer hasn't changed, Docker reuses the cached version — very fast.
If a layer changes, all subsequent layers must be rebuilt.

**Bad order (slow builds):**
```dockerfile
COPY . /app                  # copy everything
RUN pip install -r requirements.txt   # reinstall EVERY time any file changes
```

**Good order (fast builds):**
```dockerfile
COPY requirements.txt /app/  # only copy requirements file
RUN pip install -r requirements.txt   # only runs if requirements.txt changes
COPY . /app                  # copy rest of code — changing app.py doesn't redo pip install
```

---

## 11. Phase 9 — AWS Infrastructure

### What AWS services we use and why

```
┌──────────────────────────────────────────────────────────────┐
│                          GitHub                              │
│  Push to main → GitHub Actions triggers                      │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│              ECR (Elastic Container Registry)                │
│  Like DockerHub but private and inside AWS.                  │
│  GitHub Actions builds the image and pushes it here.         │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼ (EC2 pulls the image from ECR)
┌──────────────────────────────────────────────────────────────┐
│                   EC2 (Virtual Machine)                      │
│  Runs Ubuntu Linux.                                          │
│  Has Docker installed.                                       │
│  Has the GitHub Actions self-hosted runner installed.        │
│  Pulls new images from ECR and runs them as containers.      │
│                                                              │
│  Port 8081 open to the internet → users hit this            │
└──────────────────────────────────────────────────────────────┘
```

### Step 1 — IAM User (Identity and Access Management)

IAM is AWS's permissions system.
You need to create a user with permissions to:
- Push/pull from ECR
- (Optionally) stop/start EC2 instances

Go to AWS Console → IAM → Users → Create User
Attach these managed policies:
- `AmazonEC2ContainerRegistryFullAccess`
- `AmazonEC2FullAccess` (if you want GitHub Actions to start/stop EC2)

After creation, create **Access Keys** (not login password — these are for programmatic access).
Save the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` — you will need them later.

> **Security rule:** Never put AWS keys in your code or Dockerfile.
> Always pass them as environment variables or use IAM roles.

### Step 2 — ECR Repository

Go to AWS Console → ECR → Create Repository
- Visibility: Private
- Name: `waste-detection` (or whatever you want)

After creation you'll see a URI like:
`123456789.dkr.ecr.eu-north-1.amazonaws.com/waste-detection`

This is your `ECR_REGISTRY/ECR_REPOSITORY` combination.

### Step 3 — EC2 Instance

Go to AWS Console → EC2 → Launch Instance
- AMI: Ubuntu Server 22.04 LTS (free tier eligible)
- Instance type: t2.micro (free tier) or t3.medium for better performance
- Security group: open port 22 (SSH) and port 8081 (your app)
- Key pair: create/download a .pem file for SSH access

**After launching, SSH in and install Docker:**
```bash
# SSH into your EC2 instance
ssh -i your-key.pem ubuntu@<your-ec2-public-ip>

# Install Docker
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker

# Allow ubuntu user to run docker without sudo
sudo usermod -aG docker ubuntu

# Log out and log back in for group change to take effect
exit
```

### Step 4 — Self-hosted GitHub Actions Runner on EC2

GitHub Actions normally runs on GitHub's own servers ("ubuntu-latest").
But for deployment, we want it to run ON the EC2 instance so it can
directly start/stop Docker containers there.

Go to GitHub → Repository Settings → Actions → Runners → New self-hosted runner
Follow the instructions — they give you commands to run on EC2 that:
1. Download the runner agent
2. Configure it to connect to your repo
3. Start it as a background service

After this, using `runs-on: self-hosted` in your workflow will run that job on your EC2 machine.

---

## 12. Phase 10 — CI/CD Pipeline

**CI = Continuous Integration** — automatically test every push
**CD = Continuous Delivery** — automatically deploy every successful push

Without CI/CD:
- Developer writes code locally
- Manually builds Docker image
- Manually SSH into server
- Manually runs docker commands
- Prone to human error, inconsistent

With CI/CD:
- Push to GitHub
- Everything runs automatically
- Same process every time

### The full workflow file explained

```yaml
# .github/workflows/main.yaml

name: workflow

# Run this workflow whenever code is pushed to main
# (but not when only README.md changes — no point redeploying for docs)
on:
  push:
    branches:
      - main
    paths-ignore:
      - 'README.md'

# Permissions needed for GitHub's OIDC token (for AWS auth)
permissions:
  id-token: write
  contents: read

# ── Job 1: CI (runs on GitHub's servers) ──────────────────────────────────
jobs:
  integration:
    name: Continuous Integration
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v3    # downloads your repo to the runner

      - name: Lint code
        run: echo "Add real linting here (e.g., flake8 .)"

      - name: Run unit tests
        run: echo "Add real tests here (e.g., pytest tests/)"

# ── Job 2: Build image & push to ECR ──────────────────────────────────────
  build-and-push-ecr-image:
    name: Continuous Delivery
    needs: integration              # only runs if Job 1 passes
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v3

      - name: Install Utilities
        run: |
          sudo apt-get update
          sudo apt-get install -y jq unzip

      # Configure AWS credentials from GitHub Secrets
      # secrets.AWS_ACCESS_KEY_ID is set in GitHub → Settings → Secrets
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: eu-north-1

      # Log in to ECR so docker push works
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1

      # Build the Docker image and push to ECR
      - name: Build, tag, and push image to Amazon ECR
        id: build-image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: ${{ secrets.ECR_REPOSITORY_NAME }}
          IMAGE_TAG: latest
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          echo "image=$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG" >> $GITHUB_OUTPUT

# ── Job 3: Deploy on EC2 (runs on your EC2 self-hosted runner) ────────────
  Continuous-Deployment:
    needs: build-and-push-ecr-image
    runs-on: self-hosted            # this runs ON your EC2 instance
    steps:
      - name: Checkout
        uses: actions/checkout@v3

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: eu-north-1

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1

      # Stop the currently running container (ignore error if it's not running)
      - name: Stop existing container
        run: |
          docker ps -q --filter "name=waste" | grep -q . \
            && docker stop waste && docker rm -fv waste \
            || true     # "|| true" prevents failure if container doesn't exist

      # Remove old images to free up disk space
      - name: Clean old images
        run: docker system prune -af

      # Pull the latest image we just pushed
      - name: Pull latest image
        run: docker pull ${{ secrets.AWS_ECR_LOGIN_URI }}/${{ secrets.ECR_REPOSITORY_NAME }}:latest

      # Start the new container
      - name: Run new container
        run: |
          docker run -d \
            -p 8081:8081 \
            --ipc="host" \
            --name=waste \
            -e AWS_ACCESS_KEY_ID=${{ secrets.AWS_ACCESS_KEY_ID }} \
            -e AWS_SECRET_ACCESS_KEY=${{ secrets.AWS_SECRET_ACCESS_KEY }} \
            -e AWS_REGION=eu-north-1 \
            ${{ secrets.AWS_ECR_LOGIN_URI }}/${{ secrets.ECR_REPOSITORY_NAME }}:latest
```

### GitHub Secrets — what they are and how to set them

Secrets are environment variables stored encrypted in GitHub.
They are never shown in logs (GitHub automatically masks them).

Go to: GitHub → Your Repo → Settings → Secrets and Variables → Actions → New Repository Secret

| Secret name           | Value                                                  |
|-----------------------|--------------------------------------------------------|
| `AWS_ACCESS_KEY_ID`   | From the IAM user you created                          |
| `AWS_SECRET_ACCESS_KEY` | From the IAM user you created                        |
| `ECR_REPOSITORY_NAME` | `waste-detection` (or your repo name)                  |
| `AWS_ECR_LOGIN_URI`   | `123456789.dkr.ecr.eu-north-1.amazonaws.com`           |

---

## 13. Phase 11 — Logging for AWS Deployment

### Why local file logging is not enough in Docker

When a Docker container restarts, its filesystem is reset.
Any log files inside the container are GONE.

The correct approach for Docker:
1. Write logs to **stdout** (standard output)
2. Docker captures everything written to stdout
3. `docker logs waste` shows you all captured output
4. Optionally, send Docker logs to **CloudWatch Logs** for long-term retention

### The aws_logger.py we already created

```python
# wasteDetection/logger/aws_logger.py  (already written — see the file)

# Key functions:
# get_logger()        → creates a logger that writes to both stdout AND a rotating file
# log_aws_context()   → logs EC2 instance ID, AZ, container ID at startup
# run_command()       → runs shell commands and logs ALL their output (replaces os.system)
```

### How to view logs on EC2

```bash
# Live tail (follows new entries as they appear)
docker logs -f waste

# Last 100 lines
docker logs --tail 100 waste

# Since a specific time
docker logs --since 2025-01-15T10:00:00 waste

# Save to a file on the host
docker logs waste > /tmp/app-logs.txt 2>&1
```

### Optional: Send logs to AWS CloudWatch

1. On your EC2 instance, install the CloudWatch agent:
```bash
sudo apt-get install amazon-cloudwatch-agent
```

2. Configure Docker to use the `awslogs` log driver by editing `/etc/docker/daemon.json`:
```json
{
  "log-driver": "awslogs",
  "log-opts": {
    "awslogs-region": "eu-north-1",
    "awslogs-group": "/docker/waste-detection",
    "awslogs-stream": "app-logs"
  }
}
```

3. Restart Docker:
```bash
sudo systemctl restart docker
```

After this, all Docker stdout is automatically sent to CloudWatch Logs.
You can view, filter, and set alarms on them in the AWS Console.

---

## 14. How Everything Connects — End-to-End Flow

### Training flow

```
You push new training data to S3 / a URL
              ↓
Hit /train endpoint (browser or curl)
              ↓
Flask calls TrainingPipeline.run_pipeline()
              ↓
DataIngestion.initiate_data_ingestion()
  → downloads zip from URL
  → extracts to artifacts/data_ingestion/
  → returns DataIngestionArtifact
              ↓
DataValidation.initiate_data_validation()
  → checks all required folders exist
  → returns DataValidationArtifact
              ↓
ModelTrainer.initiate_model_trainer()
  → calls yolov5/train.py via subprocess
  → copies best.pt to artifacts/model_trainer/
  → copies best.pt to yolov5/best.pt
  → returns ModelTrainerArtifact
              ↓
"Training Successful" returned to caller
```

### Prediction flow

```
User's browser sends: POST /predict  { "image": "<base64>" }
              ↓
Flask predictRoute() receives request
              ↓
decode_image() writes base64 → data/inputImage.jpg
              ↓
run_command("yolov5/detect.py ...") runs detection
  → reads data/inputImage.jpg
  → draws bounding boxes on detected waste
  → saves result to yolov5/runs/detect/exp/inputImage.jpg
              ↓
encode_image() reads result → base64
              ↓
Flask returns: { "image": "<base64 of annotated image>" }
              ↓
User's browser displays the annotated image
```

### Deployment flow (triggered by git push)

```
git push origin main
              ↓
GitHub Actions detects push to main
              ↓
Job 1 (Integration): lint + tests pass
              ↓
Job 2 (Build): docker build + push to ECR
              ↓
Job 3 (Deploy): runs on EC2 self-hosted runner
  → docker stop waste (old container)
  → docker system prune (clean up)
  → docker pull (get new image from ECR)
  → docker run (start new container)
              ↓
Users now hit the new version at <EC2-IP>:8081
```

---

## 15. Common Mistakes & How to Avoid Them

### 1. `os.system()` swallows errors silently

```python
# BAD: if detect.py crashes, you'll never know why
os.system("python detect.py")

# GOOD: capture output, check return code, log everything
result = subprocess.run("python detect.py", shell=True, capture_output=True, text=True)
if result.returncode != 0:
    logger.error("detect.py failed:\n%s", result.stderr)
    raise Exception("Detection failed")
logger.info(result.stdout)
```

### 2. Flask running on localhost inside Docker

```python
# BAD: no traffic can enter the container
app.run(host="127.0.0.1", port=8081)

# GOOD: accepts traffic from outside the container
app.run(host="0.0.0.0", port=8081)
```

### 3. Hard-coding paths that break in Docker

```python
# BAD: works on Windows, breaks in Docker (Linux)
path = "C:\\Users\\you\\project\\data\\image.jpg"

# GOOD: relative paths work everywhere
path = "data/inputImage.jpg"

# ALSO GOOD: from_root() finds the project root reliably
from from_root import from_root
path = os.path.join(from_root(), "data", "inputImage.jpg")
```

### 4. Not cleaning up YOLOv5 run directories

YOLOv5 saves output to `yolov5/runs/detect/exp/`.
On the second call it creates `exp2/`, then `exp3/`, etc.
Eventually your disk fills up.

```python
# Always clean up after each prediction
run_command("rm -rf yolov5/runs", logger)
```

### 5. Docker layer caching not being used

```dockerfile
# SLOW: copy all files first, then install (pip install re-runs when ANY file changes)
COPY . /app
RUN pip install -r requirements.txt

# FAST: install deps first (pip install only re-runs when requirements.txt changes)
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app                # changing app.py doesn't redo pip install
```

### 6. Forgetting that Docker containers are ephemeral

Everything inside a container is **lost when it stops** unless:
- You **mount a volume**: `-v /host/path:/container/path`
- You copy files out before stopping

For ML models: copy the trained `best.pt` to S3 or EFS.
For logs: stream to CloudWatch, not just local files.

### 7. Putting secrets in code or Dockerfile

```dockerfile
# NEVER DO THIS — AWS keys in the image = security breach
ENV AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
```

```python
# BAD — secret in code
S3_BUCKET = "my-bucket"
AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
```

```python
# GOOD — read from environment variable, set at runtime
import os
AWS_KEY = os.environ["AWS_ACCESS_KEY_ID"]   # set via docker run -e or GitHub Secrets
```

### 8. Not handling the case where detection output is missing

YOLOv5 only creates `runs/detect/exp/` if it finds at least one object.
If nothing is detected, your `encode_image()` call will crash with FileNotFoundError.

```python
result_path = "yolov5/runs/detect/exp/inputImage.jpg"
if not os.path.exists(result_path):
    logger.warning("No objects detected — returning original image")
    result_path = "data/inputImage.jpg"   # return original if nothing found

result_b64 = encode_image(result_path)
```

---

## Quick Reference — Commands You'll Use All The Time

```bash
# ── Python package ─────────────────────────────────────────────────────────
pip install -e .                          # install package in editable mode
pip install -r requirements.txt           # install all dependencies

# ── Docker ─────────────────────────────────────────────────────────────────
docker build -t waste-detection:latest .  # build image
docker run -d -p 8081:8081 --name waste waste-detection:latest  # run container
docker logs -f waste                      # tail logs
docker exec -it waste bash                # open a shell inside running container
docker stop waste && docker rm waste      # stop and remove container
docker images                             # list all images on machine
docker system prune -af                   # remove ALL stopped containers and images

# ── AWS ECR ───────────────────────────────────────────────────────────────
# Login to ECR (replace region and account ID)
aws ecr get-login-password --region eu-north-1 | \
  docker login --username AWS --password-stdin 123456789.dkr.ecr.eu-north-1.amazonaws.com

docker tag waste-detection:latest 123456789.dkr.ecr.eu-north-1.amazonaws.com/waste-detection:latest
docker push 123456789.dkr.ecr.eu-north-1.amazonaws.com/waste-detection:latest

# ── Git ───────────────────────────────────────────────────────────────────
git add .
git commit -m "feat: add data ingestion component"
git push origin main                      # triggers the CI/CD pipeline
```

---

## Suggested Build Order (Week by Week)

| Week | What to build                                      | Goal                            |
|------|----------------------------------------------------|---------------------------------|
| 1    | setup.py, requirements.txt, folder structure       | Project skeleton works          |
| 1    | logger/__init__.py, exceptions/__init__.py         | Errors are visible and traceable|
| 2    | constants, config_entity, artifact_entity          | Configuration is centralised    |
| 2    | data_ingestion.py                                  | Can download & unzip data       |
| 3    | data_validation.py                                 | Bad data is caught early        |
| 3    | model_trainer.py                                   | Can train YOLOv5 on the data    |
| 4    | training_pipeline.py                               | Can run full pipeline in 1 call |
| 4    | app.py + templates/index.html                      | Flask app works locally         |
| 5    | Dockerfile                                         | App runs in Docker locally      |
| 5    | AWS setup (IAM, ECR, EC2)                          | Infrastructure ready            |
| 6    | .github/workflows/main.yaml                        | Auto-deploys on every push      |
| 6    | aws_logger.py                                      | Logs visible in docker logs     |

---

*Remember: the best way to learn is to build it yourself, break it, debug it,
and build it again. Each cycle makes the concepts permanently clear.
Don't rush to have it working — rush to understand why it works.*
