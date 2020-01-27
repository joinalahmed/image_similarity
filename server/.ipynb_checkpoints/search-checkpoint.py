import random
import tensorflow as tf
import numpy as np
import os
import scipy.io
import time
from datetime import datetime
from scipy import ndimage
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors
import pickle 
from PIL import Image
import gc
import os
import cv2
from tempfile import TemporaryFile
from tensorflow.python.platform import gfile
import glob
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1


def get_top_k_similar(image_data, pred, pred_final,imagePath,k=1):
        print("total data",len(pred))
        print(image_data.shape)
        names=[]
        top_k_ind = np.argsort([cosine(image_data, pred_row) for ith_row, pred_row in enumerate(pred)])
        count=0
        for i, neighbor in enumerate(top_k_ind[:k]):
            key="/".join(pred_final[neighbor].split('/')[:-1])
            files=glob.glob(key+'/*.jpeg')
            for f in files:
                if f == pred_final[neighbor]:
                    continue
                count+=1
                
                image = ndimage.imread(f)
                timestr = datetime.now().strftime("%Y%m%d%H%M%S")
                name= timestr+"."+str(count)
                name2 = 'static/result/image'+"_"+name+'.jpg'
                h,w,c=image.shape
                if (h/w)==0.6 or h<4048:
                    continue
                cv2.imwrite(name2,cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                name1 = 'result/image'+"_"+name+'.jpg'
                
                names.append(name1)
        print("len",len(names))
        if len(names) < 9:
            diff = 9-len(names)
            print("diff",diff)
            for i, neighbor in enumerate(top_k_ind[1:diff+1]):
                count+=1
                key=pred_final[neighbor]
                image = ndimage.imread(key)
                timestr = datetime.now().strftime("%Y%m%d%H%M%S")
                name= timestr+"."+str(count)
                name2 = 'static/result/image'+"_"+name+'.jpg'
                cv2.imwrite(name2,cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                name1 = 'result/image'+"_"+name+'.jpg'
                names.append(name1)
        print("len",len(names))
        return names
                
def create_inception_graph():
    with tf.Session() as sess:
        model_filename = os.path.join(
            'imagenet', 'classify_image_graph_def.pb')
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
              tf.import_graph_def(graph_def, name='', return_elements=[
                  BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                  RESIZED_INPUT_TENSOR_NAME]))
        return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor

def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor,{image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values        

def recommend(imagePath, extracted_features):
    tf.reset_default_graph()
    config = tf.ConfigProto(device_count = {'GPU': 0})
    sess = tf.Session(config=config)
    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (create_inception_graph())
    image_data = gfile.FastGFile(imagePath, 'rb').read()
    features = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)	
    names=[]
    with open('../lib/neighbor_list_recom.pickle','rb') as f:
        neighbor_list = pickle.load(f) 
        names=get_top_k_similar(features, extracted_features, neighbor_list,imagePath, k=1)
    return names

