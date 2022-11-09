import cv2
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from keras.applications.vgg16 import preprocess_input
import tensorflow as tf

model = tf.keras.models.load_model('model.h5')


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


IMG_SIZE = (180, 180)


def getResult(image, v):
    image = preprocess_imgs(image, IMG_SIZE)
    # print(image.shape)
    result = v+model.predict(image)[0][0]
    return result
