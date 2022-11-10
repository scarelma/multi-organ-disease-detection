# Brain server
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import flask
from flask import request, jsonify
import numpy as np
import tensorflow as tf
import imutils
import cv2
from keras.applications.vgg16 import preprocess_input

app = flask.Flask(__name__)

model = tf.keras.models.load_model('model.h5')

IMG_SIZE = (224, 224)


def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS,
                      extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)

    return np.array(set_new)


def preprocess_imgs(set_name, img_size):
    """
    Resize and apply VGG-15 preprocessing
    """
    set_new = []
    for img in set_name:
        img = cv2.resize(
            img,
            dsize=img_size,
            interpolation=cv2.INTER_CUBIC
        )
        set_new.append(preprocess_input(img))
    return np.array(set_new)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "GET":
        return "kindly use POST method"
    if request.method == "POST":
        recievedFile = request.files['fileUpload']
        print('File Recieved')
        if recievedFile.filename != '':
            recievedFile.save(str(recievedFile.filename))
            image_path = str(recievedFile.filename)
            image = cv2.imread(image_path)
            image = crop_imgs([image])
            # print(image.shape)
            image = preprocess_imgs(image, IMG_SIZE)
            result = model.predict(image)
            os.remove(image_path)
            if result[0][0] > 0.5:
                return jsonify({'result': 'yes'})
            return jsonify({'result': 'no'})
        else:
            return "No file recieved"


@app.route('/info', methods=['GET'])
def info():
    return '''<h1>Brain Server</h1>
<p>Brain Server is a Flask API for Brain.</p>'''


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5001)

# image = cv2.imread('1 no.jpeg')
# # image = preprocess(image)
# print(image.shape)
# image = crop_imgs([image])
# print(image.shape)
# image = preprocess_imgs(image, IMG_SIZE)
# print(image.shape)
# result = model.predict(image)
# print(result)
