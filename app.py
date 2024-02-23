from flask import Flask,request, url_for, redirect, render_template
import pandas as pd
import numpy as np
import pickle
import sqlite3
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import numpy as np
from keras.layers import *
from PIL import Image
import matplotlib.pyplot as plt
from DataReader import DataReader
import cv2
from skimage.metrics import structural_similarity as ssim
from math import log10, sqrt
import os
from keras.models import model_from_json
from util.data_utils import getPaths, read_and_resize, preprocess, deprocess
from werkzeug.utils import secure_filename
import tensorflow as tf
import argparse
import io
import os
from PIL import Image
import cv2
import numpy as np
from torchvision.models import detection
import sqlite3
import torch
from torchvision import models

app = Flask(__name__)

model = torch.hub.load("ultralytics/yolov5", "custom", path = "best.pt", force_reload=True)

model.eval()
model.conf = 0.5  
model.iou = 0.45  

from io import BytesIO

def gen():
    """
    The function takes in a video stream from the webcam, runs it through the model, and returns the
    output of the model as a video stream
    """
    cap=cv2.VideoCapture(0)
    while(cap.isOpened()):
        success, frame = cap.read()
        if success == True:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            img = Image.open(io.BytesIO(frame))
            results = model(img, size=640)
            results.print()  
            img = np.squeeze(results.render()) 
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
        else:
            break
        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    """
    It returns a response object that contains a generator function that yields a sequence of images
    :return: A response object with the gen() function as the body.
    """
    return Response(gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

UPLOAD_FOLDER = 'static/uploads/'

app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def PSNR(original, compressed):
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)  
    mse_value = np.mean((original - compressed) ** 2) 
    if(mse_value == 0):
        return 100
    max_pixel = 255.0
    psnr_value = 100 - (20 * log10(max_pixel / sqrt(mse_value))) 
    return psnr_value / 2

def imageSSIM(normal, embed):
    normal = cv2.cvtColor(normal, cv2.COLOR_BGR2GRAY)
    embed = cv2.cvtColor(embed, cv2.COLOR_BGR2GRAY) 
    ssim_value = ssim(normal, embed, data_range = embed.max() - embed.min())
    return ssim_value

graph = tf.get_default_graph()

tf.reset_default_graph()

from tensorflow.keras.layers import Conv2D, ReLU, Concatenate

def getMultpatchModel(RGB):
    cnn1 = Conv2D(3, (1, 1), padding="same", activation="relu", use_bias=True,
                  kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))(RGB)  # layer 1

    cnn2 = Conv2D(3, (3, 3), padding="same", activation="relu", use_bias=True,
                  kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))(cnn1)  # layer 2

    fa_block1 = Concatenate(axis=-1)([cnn1, cnn2])  # concatenate layer1 and layer2 to form a residual network

    cnn3 = Conv2D(3, (5, 5), padding="same", activation="relu", use_bias=True,
                  kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))(fa_block1)

    fa_block2 = Concatenate(axis=-1)([cnn2, cnn3])  # concatenate layer2 and layer3 to form a residual network

    cnn4 = Conv2D(3, (7, 7), padding="same", activation="relu", use_bias=True,
                  kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))(fa_block2)

    CA = Concatenate(axis=-1)([cnn1, cnn2, cnn3, cnn4])

    cnn5 = Conv2D(3, (3, 3), padding="same", activation="relu", use_bias=True,
                  kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))(CA)

    MAX = cnn5  # max layer

    multipatch = ReLU(max_value=1.0)(tf.math.multiply(MAX, RGB) - MAX + 1.0)  # replace pixels intensity

    return multipatch



RGB = tf.placeholder(shape=(None,400, 400,3),dtype=tf.float32)
MAX = tf.placeholder(shape=(None,400, 400,3),dtype=tf.float32)
multi_patch_model = getMultpatchModel(RGB) #loading and generating multi patch model
trainingLoss = tf.reduce_mean(tf.square(multi_patch_model-MAX)) #optimizations
optimizerRate = tf.train.AdamOptimizer(1e-4)
trainVariables = tf.trainable_variables()
gradient = optimizerRate.compute_gradients(trainingLoss,trainVariables)
clippedGradients = [(tf.clip_by_norm(gradients,0.1),var1) for gradients,var1 in gradient]
optimize = optimizerRate.apply_gradients(gradient)
saver = tf.train.Saver()
with open('models/gen_p/model_15320_.json', "r") as json_file:
    loaded_model_json = json_file.read()
json_file.close()    
multi_patch_model = model_from_json(loaded_model_json)
multi_patch_model.load_weights('models/gen_p/model_15320_.h5')
orig = cv2.imread("Dataset/reference-890/100_img_.png")
height, width, channels = orig.shape
orig = cv2.resize(orig,(256, 256),interpolation = cv2.INTER_CUBIC)
inp_img = read_and_resize("Dataset/raw-890/100_img_.png", (256, 256))
im = preprocess(inp_img)
im = np.expand_dims(im, axis=0) 
gen = multi_patch_model.predict(im)
enhance_Image = deprocess(gen)[0]
propose_psnr = PSNR(orig, enhance_Image)
propose_ssim = imageSSIM(orig, enhance_Image)
print("PSNR Value " + str(propose_psnr))
print("SSIM Value " + str(propose_ssim))




@app.route('/')
def hello_world():
    return render_template("home.html")

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route("/signup")
def signup():

    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
    con.commit()
    con.close()
    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("index.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signup.html")

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save('static/test.jpg')

        inp_img = read_and_resize('static/test.jpg', (256, 256))
        im = preprocess(inp_img)
        im = np.expand_dims(im, axis=0) 
        enhance_Image = multi_patch_model.predict(im)
        enhance_Image = deprocess(enhance_Image)[0]
        enhance_Image = cv2.resize(enhance_Image, (256, 256), interpolation=cv2.INTER_CUBIC)
        enhance_Image = cv2.cvtColor(enhance_Image, cv2.COLOR_BGR2RGB)
        
        # Save enhanced image as 'test1.jpg'
        cv2.imwrite('static/test1.jpg', cv2.cvtColor(enhance_Image, cv2.COLOR_RGB2BGR))

        orig = cv2.imread('static/test.jpg')
        orig = cv2.resize(orig, (256, 256), interpolation=cv2.INTER_CUBIC)
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

        file = cv2.imread('static/test1.jpg')
        img = Image.fromarray(cv2.cvtColor(file, cv2.COLOR_BGR2RGB))
        
        results = model(img, size=640)
        results.render()  
        for img in results.render():
            img_base64 = Image.fromarray(img)
            img_base64.save("static/test2.jpg", format="JPEG")

        return render_template('result.html', filename=filename, orig='test.jpg', enhance_Image='test1.jpg',detection='test2.jpg')

    return redirect(request.url)




@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/notebook')
def notebook():
	return render_template('notebook.html')

@app.route('/notebook1')
def notebook1():
	return render_template('notebook1.html')


if __name__ == '__main__':
    app.run(debug=False, threaded=False)
