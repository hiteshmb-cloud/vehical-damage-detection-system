import os
import shutil
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from ultralytics import YOLO
from flask import Flask, request, render_template

import warnings
warnings.filterwarnings('ignore')

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
inputimgpath = './static/images/inputimg.jpg'
outputimgpath = './static/images/runs/detect/predict/inputimg.jpg'


app = Flask(__name__)

ALLOWED_EXTENSIONS = {'jpg'}


def allowed_file(filename):
  return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index(): 
  return render_template('index.html')


@app.route('/home')
def home(): 
  return render_template('index.html')


@app.route('/vddsservice')
def vddsservice():
  return render_template('service.html',condval="nooutput")


@app.route('/vehicledmgdetection',methods=['POST'])
def vehicledmgdetection():
  shutil.rmtree(r'./runs', ignore_errors=True)
  file = None
  file = request.files['file']
  if file and allowed_file(file.filename):

    image = Image.open(file)
    image.save(os.path.join(inputimgpath))
    
    os.system("yolo task=detect mode=predict model=./weights/best.pt source=./static/images/inputimg.jpg conf=0.5")
    shutil.copy(src='./runs/detect/predict/inputimg.jpg', dst="./static/images/runs/detect/predict/inputimg.jpg") 

  return render_template('service.html', outputimgpath = outputimgpath)


if __name__ == '__main__':
    app.run(debug=False)



