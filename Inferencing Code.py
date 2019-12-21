import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.platform import gfile
from PIL import Image
import numpy as np
import time
from matplotlib import pyplot as plt
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import os
from numpy import array
import time

output_names = ['activation_1/Softmax']
input_names = ['input_1']

def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


trt_graph = get_frozen_graph('./model/TensorRT_Model.pb')

# Create session and load graph
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')


# Get graph input size
for node in trt_graph.node:
    if 'input_' in node.name:
        size = node.attr['shape'].shape
        image_size = [size.dim[i].size for i in range(1, 4)]
        break
print("image_size: {}".format(image_size))

# input and output tensor names.
input_tensor_name = input_names[0] + ":0"
output_tensor_name = output_names[0] + ":0"

print("input_tensor_name: {}\noutput_tensor_name: {}".format(
    input_tensor_name, output_tensor_name))

output_tensor = tf_sess.graph.get_tensor_by_name(output_tensor_name)

path = './Test/Normal_Videos_041_x264'
i = 0
times = []
images = []

for file in os.listdir(path):
  if i < 16:
    start_time = time.time()
    img = image.load_img(path+ '/'+ file, target_size=image_size[:2])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images.append(x)
    i += 1
  else:
    x = np.vstack(images)
    x = array(x).reshape(1, 112, 112, 16, 3)
    x = preprocess_input(x)
    images = []

    feed_dict = {
      input_tensor_name: x
    }
    preds = tf_sess.run(output_tensor, feed_dict)
    delta = (time.time() - start_time)
    times.append(delta)
    print("Needed time in Inferece: ", delta)
    i = 0

mean_delta = np.array(times).mean()
fps = 1 / delta
print('average(sec):{:.4f},fps:{:.2f}'.format(mean_delta, fps))