# -*- coding: utf-8 -*-

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='model_inception.h5'

# Load your trained model
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
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="Apple_black_rot"
    elif preds==1:
        preds="Apple_healthy"
    elif preds==2:
        preds="Apple_scab"
    elif preds==3:
        preds="Cassava_bacterial_blight"
    elif preds==4:
        preds="Cassava_brown_streak_disease"
    elif preds==5:
        preds="Cassava_green_mottle"
    elif preds==6:
        preds="Cassava_healthy"
    elif preds==7:
        preds="Cassava_mosaic_disease"
    elif preds==8:
        preds="Cherry_health"
    elif preds==9:
        preds="Cherry_powdery_mildew"
    elif preds==10:
        preds="Chilli_healthy"
    elif preds==11:
        preds="Chilli_leaf curl"
    elif preds==12:
        preds="Chilli_leaf spot"
    elif preds==13:
        preds="Chilli_whitefly"
    elif preds==14:
        preds="Chilli_yellowish"
    elif preds==15:
        preds="Coffee_cercospora_leaf_spot"
    elif preds==16:
        preds="Coffee_healthy"
    elif preds==17:
        preds="Coffee_red_spider_mite"
    elif preds==18:
        preds="Coffee_rust"
    elif preds==19:
        preds="Corn_common_rust"
    elif preds==20:
        preds="Corn_gray_leaf_spot"
    elif preds==21:
        preds="Corn_healthy"
    elif preds==22:
        preds="corn_northern_leaf_blight"
    elif preds==23:
        preds="Cucumber_diseased"
    elif preds==24:
        preds="Cucumber_healthy"
    elif preds==25:
        preds="Gauva_disease"
    elif preds==26:
        preds="Gauva_healthy"
    elif preds==27:
        preds="Grape_black_measles"
    elif preds==28:
        preds="Grape_black_rot"
    elif preds==29:
        preds="Grape_healthy"
    elif preds==30:
        preds="Grape_leaf_blight_(isariopsis_leaf_spot)"
    elif preds==31:
        preds="Jamun_diseased"
    elif preds==32:
        preds="Jamun_healthy"
    elif preds==33:
        preds="Lemon_disease"
    elif preds==34:
        preds="Mango_healthy"
    elif preds==35:
        preds="Mango_disease"
    elif preds==36:
        preds="Peach_bacterial_spot"
    elif preds==37:
        preds="Peach_healthy"
    elif preds==38:
        preds="Pepper_bell_bacterial_spot"
    elif preds==39:
        preds="Pepper_bell_healthy"
    elif preds==40:
        preds="Pomegranate_diseased"
    elif preds==41:
        preds="Pomegranate_healthy"
    elif preds==42:
        preds="Potato_early_blight"
    elif preds==43:
        preds="Potato_healthy"
    elif preds==44:
        preds="Potato_leaf_blast"
    elif preds==45:
        preds="Rice_neck_blast"
    elif preds==46:
        preds="Rice_healthy"
    elif preds==47:
        preds="Rice_late_blight"
    elif preds==48:
        preds="Rice_brown_spot"
    elif preds==49:
        preds="Rice_hispa"
    elif preds==50:
        preds="Soybean_bacterial_blight"
    elif preds==51:
        preds="Soybean_caterpillar"
    elif preds==52:
        preds="Soybean_diabrotica_speciosa"
    elif preds==53:
        preds="Soybean_downy_mildew"
    elif preds==54:
        preds="Soybean_healthy"
    elif preds==55:
        preds="Soybean_mosaic_virus"
    elif preds==56:
        preds="Soybean_powdery_mildew"
    elif preds==57:
        preds="Soybean_rust"
    elif preds==58:
        preds="Soybean_southern_blight"
    elif preds==59:
        preds="Strawberry_leaf_scorch"
    elif preds==60:
        preds="Strawberry_healthy"
    elif preds==61:
        preds="Sugarcane_bacterial_blight"
    elif preds==62:
        preds="Sugarcane_healthy"
    elif preds==63:
        preds="Sugarcane_red_rot"
    elif preds==64:
        preds="Sugarcane_red_stripe"
    elif preds==65:
        preds="Sugarcane_rust"
    elif preds==66:
        preds="Tea_algal_leaf"
    elif preds==67:
        preds="Tea_anthracnose"
    elif preds==68:
        preds="Tea_bird_eye_spot"
    elif preds==69:
        preds="Tea_brown_blight"
    elif preds==70:
        preds="Tea_healthy"
    elif preds==71:
        preds="Tea_red_leaf_spot"
    elif preds==72:
        preds="Tomato_bacterial_spot"
    elif preds==73:
        preds="Tomato_early_blight"
    elif preds==74:
        preds="Tomato_healthy"
    elif preds==75:
        preds="Tomato_late_blight"
    elif preds==76:
        preds="Tomato_leaf_mold"
    elif preds==77:
        preds="Tomato_mosaic_virus"
    elif preds==78:
        preds="Tomato_septoria_leaf_spot"
    elif preds==79:
        preds="Tomato_spider_mites_(two_spotted_spider_mite)"
    elif preds==80:
        preds="Tomato_target_spot"
    elif preds==81:
        preds="Tomato_yellow_leaf_curl_virus"
    elif preds==82:
        preds="Wheat_brown_rust"
    elif preds==83:
        preds="Wheat_healthy"
    elif preds==84:
        preds="Wheat_septoria"
    elif preds==85:
        preds="Wheat_yellow_rust"
    elif preds==86:
        preds="Apple_rust"
    elif preds==87:
        preds="Lemon_healthy"
    else:
        preds="Healthy"
        
    
    
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
    return ok


if __name__ == '__main__':
    app.run(port=5001,debug=True)
