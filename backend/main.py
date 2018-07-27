from flask import Flask, jsonify
from flask import request
from flask_cors import CORS, cross_origin
import png
import numpy
import numpy as np
import ocr
import os
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import color
UPLOAD_FOLDER = 'upload_folder'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)


def analyze(fname):
    image = mpimg.imread(UPLOAD_FOLDER + "/" + fname)
    value = ocr.predict(image)
    responseObject = dict()
    responseObject["value"] = [value]
    print(responseObject)
    response = jsonify(responseObject)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
    
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/analyzeUpload', methods=['GET', 'POST'])
def upload_file():
    print(request)
    if request.method == 'POST':
        print(request.files)
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            response = jsonify("No file found")
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('No selected file')
            response = jsonify("No file found")
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        if file and allowed_file(file.filename):
            # filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            time.sleep(1)
            return analyze(file.filename)
            
            
    response = jsonify("Upload failed")
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response