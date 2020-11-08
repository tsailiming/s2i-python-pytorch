#!/usr/bin/env python

from flask import Flask, request, jsonify
from flask_cors import CORS

#import imageio
from PIL import Image, ImageOps

#from scipy.misc import imsave,imread, imresize
#import numpy as np

#import keras.models
import re
import base64
from io import BytesIO

import sys 
import os

import torch
from torchvision import transforms

from mnist.net import Net

#sys.path.append(os.path.abspath("./model"))
#from load import *

app = Flask(__name__)
CORS(app)

#global model, graph
#model, graph = init()

model = Net()
model.load_state_dict(torch.load(os.getenv('model_path', './model/mnist_cnn.pt')))
model.eval()

#@app.route('/')
#def index():
#    return render_template("index.html")

@app.route('/predict/', methods=['GET','POST'])
def predict():
    # get data from drawing canvas and save as image
    img = parseImage(request.get_data())

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    result  = model.forward(transform(img).unsqueeze(0)) #.max(1)
    num = result.max(1).indices.item()
    return jsonify({'result': num, 'data': result.tolist()[0]})
    
    
def parseImage(imgData):
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    return ImageOps.invert(Image.open(BytesIO(base64.b64decode(imgstr))).convert('L')).resize((28,28))

if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)