
import os
from flask import request, jsonify
from src.helper import save_img
import cv2
from src.image import getResult


def home():
    if request.method == "GET":
        return "kindly use POST method"
    if request.method == "POST":
        recievedFile = request.files['fileUpload']
        print('File Recieved')
        if recievedFile.filename != '':
            filename, v = save_img(recievedFile)
            image = [cv2.imread(filename)]
            result = getResult(image, v)
            os.remove(filename)
            if result < 0.5:
                return jsonify({'result': 'yes'})
            return jsonify({'result': 'no'})
        else:
            return "No file recieved"


def info():
    return '''<h1>Lung Server</h1>
<p>Lung Server is a Flask API for Lung.</p>'''
