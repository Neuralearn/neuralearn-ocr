import os
import logging
from logging import Formatter, FileHandler
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from module import OCRDet
import cv2
import numpy as np
from PIL import Image
import io
####from ocr import process_image


app = Flask(__name__)
_VERSION = 1  # API version


UPLOAD_FOLDER = './data/images/'

@app.route('/', methods = ["GET","POST"])
def main():

    if request.method == 'POST':
        file = request.files['file']
        image_bytes = file.read()

        image = np.asarray(bytearray(image_bytes), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        return {'output':OCRDet().pred(image)}

    return {'hello':'hello'}
    
@app.errorhandler(500)
def internal_error(error):
    print(str(error))  # ghetto logging


@app.errorhandler(404)
def not_found_error(error):
    print(str(error))

# if not app.debug:
#     file_handler = FileHandler('error.log')
#     file_handler.setFormatter(
#         Formatter('%(asctime)s %(levelname)s: \
#             %(message)s [in %(pathname)s:%(lineno)d]')
#     )
#     app.logger.setLevel(logging.INFO)
#     file_handler.setLevel(logging.INFO)
#     app.logger.addHandler(file_handler)
#     app.logger.info('errors')


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug = False)