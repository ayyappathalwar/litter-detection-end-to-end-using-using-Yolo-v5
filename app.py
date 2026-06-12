from WasteDetection.logger import logging
from WasteDetection.exceptions import AppException
import sys

try:
    a = 3/"s"

except Exception as e:   
    raise AppException(e, sys)