from flask import Flask, jsonify, abort, request, make_response, url_for,redirect, render_template
from flask_httpauth import HTTPBasicAuth
from werkzeug.utils import secure_filename
import os
import sys
sys.path.append('../')
import shutil 
import numpy as np
from search import recommend
import tarfile
from datetime import datetime
from scipy import ndimage
from scipy.misc import imsave 
UPLOAD_FOLDER = 'uploads'
from tensorflow.python.platform import gfile
app = Flask(__name__, static_url_path = "")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
auth = HTTPBasicAuth()

extracted_features=np.zeros((10000,2048),dtype=np.float32)
with open('saved_features_recom.txt') as f:
    for i,line in enumerate(f):
        extracted_features[i,:]=line.split()
        
print("loaded extracted_features") 


@app.route('/imgUpload', methods=['GET', 'POST'])
def upload_img():
    print("image upload")
    result = 'static/result'
    if not gfile.Exists(result):
          os.mkdir(result)
 
    if request.method == 'POST' or request.method == 'GET':
        if 'file' not in request.files:
           
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file :
         
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            inputloc = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_list=recommend(inputloc, extracted_features)
            image_path = "/result"
            print(image_list)

            images = {
                'image0':image_list[0],
                'image1':image_list[1],
                'image2':image_list[2],
                'image3':image_list[3],
                'image4':image_list[4],
                'image5':image_list[5],
                'image6':image_list[6],
                'image7':image_list[7],
                'image8':image_list[8]
            }
            return jsonify(images)

@app.route("/")
def main():
    return render_template("main.html")

if __name__ == '__main__':
    app.run(debug = True, host= '0.0.0.0')
