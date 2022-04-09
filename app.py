# -*- coding: utf-8 -*-
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
app = Flask(__name__)

MODEL_PATH ='model1_vgg16.h5'

model = load_model(MODEL_PATH)


def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   
    preds = model.predict(x)
    print(preds)
    preds=np.argmax(preds, axis=1)
    print(preds)
    if preds==0:
        preds="The Disease is Bacterial spot || हा रोग बॅक्टेरियल स्पॉट आहे ||  रोग बैक्टीरियल स्पॉट है"
    elif preds==1:
        preds="The Disease is Early blight || हा रोग अर्ली बलाइट  आहे || रोग अर्ली बलाइट है"
    elif preds==2:
        preds="The Disease is Tomato Late blight || हा रोग टोमॅटो लेट बलाइट आहे  || रोग टमाटर लेट बलाइट है"
    elif preds==3:
        preds="The Disease is Tomato Leaf Mold || हा रोग टोमॅटो लीफ़ मोल्ड आहे || रोग टमाटर का पत्ता मोल्ड है"
    elif preds==4:
        preds="The Disease is Tomato Septoria leaf spot || हा रोग टोमॅटो सेप्टोरिया लीफ़ स्पॉट आहे || रोग टोमेटो सेप्टोरिया लीफ़ स्पॉट है"
    elif preds==5:
        preds="The Disease is Tomato Spider mites || हा रोग म्हणजे टोमॅटो स्पायडर माइट्स || रोग टमाटर स्पाइडर माइट्स है"
    elif preds==6:
        preds="The Disease is Tomato Target Spot || हा रोग टोमॅटो टारगेट स्पॉट आहे || रोग टमाटर टारगेट स्पॉट है "
    elif preds==7:
        preds="The Disease is Tomato Yellow Leaf Curl Virus || हा रोग म्हणजे टोमॅटो येलो लीफ कर्ल व्हायरस || रोग टोमेटो येलो लीफ कर्ल वायरस है"
    elif preds==8:
        preds="The Disease is Tomato mosaic virus || हा रोग टोमॅटो मोज़ेक व्हायरस आहे || रोग टमाटर मोज़ेक वायरस है"
    elif preds==9:
        preds="The Tomato is healthy || टोमॅटो निरोगी आहे  || टमाटर सेहतमंद है"
   
    
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=True)
