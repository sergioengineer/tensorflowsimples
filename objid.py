#runfile('E:\\TensorFlow\\research\\object_detection\\utils\\label_map_util_test.py', wdir='E:\\TensorFlow\\research\\object_detection\\utils')
#runfile('E:\\TensorFlow\\research\\object_detection\\builders\\model_builder_test.py', wdir='E:\\TensorFlow\\research\\object_detection\\builders')
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


#nvidia-smi
#nvcc --version

import os

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #makes TF ignore all GPU related warnings
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
#import tensorflow as tf
import numpy as np
import zipfile
import time
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
 
import cv2

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
#sess = tf.Session(config=config)
#tf.enable_eager_execution(config=config)
#session = tf.InteractiveSession(config=config)
 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

sys.path.append("..")

from utils import label_map_util
 
from utils import visualization_utils as vis_util

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print("CHECKPOINT1")

#with tf.device("/cpu:0"):
MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
#MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
 
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
 
# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = os.path.join('E:\\TensorFlow\\research\\object_detection\\data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
print("CHECKPOINT2")
#tf.disable_v2_behavior() 
#opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
#tar_file = tarfile.open(MODEL_FILE)
#for file in tar_file.getmembers():
#  file_name = os.path.basename(file.name)
#  if 'frozen_inference_graph.pb' in file_name:
#    tar_file.extract(file, os.getcwd())
print("CHECKPOINT3")

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
print(PATH_TO_LABELS)
time.sleep(3)
#label_map_util breaking kernel
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

cap = cv2.VideoCapture("http://192.168.0.103:4747/video/mjpegfeed?640x480")
print("CHECKPOINT5 " + tf.__version__)
with detection_graph.as_default():
  with tf.compat.v1.Session(config = config, graph=detection_graph) as sess:
    while True:
        ret, image_np = cap.read()
        # expandir
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # .
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
              use_normalized_coordinates=True,
            line_thickness=8)
        print("AA")
        cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows() 
            break