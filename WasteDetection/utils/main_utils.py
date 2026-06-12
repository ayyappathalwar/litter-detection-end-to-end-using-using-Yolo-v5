import os.path
import sys
import yaml
import base64


from WasteDetection.exceptions import AppException
from WasteDetection.logger import logging


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            logging.info(f"Reading yaml file: {file_path}")
            return yaml.safe_load(yaml_file)
    
    except Exception as e:
        raise AppException(e, sys) from e
    

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                logging.info(f"Replacing existing yaml file: {file_path}")
                os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as yaml_file:
            yaml.dump(content, yaml_file)
            logging.info(f"Writing content to yaml file: {file_path}")
        
    except Exception as e:
        raise AppException(e, sys) from e
    



def decode_image(imgstring, filename):
    imgdata = base64.b64decode(imgstring)
    with open("./data/" + filename, "wb") as f:
        f.write(imgdata)
        f.close()


def encode_image(cropped_image_path):
    with open(cropped_image_path, "rb") as f:
        return base64.b64encode(f.read())
    


    