
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# check version
if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


# imports from the object detection module.
from utils import label_map_util
from utils import visualization_utils as vis_util


############## Model preparation ##############

# model selection
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17' # one of the model with SSD + MOBILENET + COCO dataset
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb' #  " "Freeze" the actual model that is used for the object detection.

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90 # number of class in .pbtxt



############## Download Model ##############
"""
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)

tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())
"""

############## Load a (frozen) Tensorflow model into memory ##############

# tf.graph
# A Graph contains a set of tf.Operation objects, which represent units of computation; 
# and tf.Tensor objects, which represent the units of data that flow between operations.

### set detection_graph as default =  forzen model
detection_graph = tf.Graph()
with detection_graph.as_default(): 
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='') # imports the graph from graph_def into the current default Graph


############## Loading label map (.pbtxt)##############

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


############## Detection ##############

import cv2
cap = cv2.VideoCapture(0)

cap.set(3,1024)
cap.set(4,768)
cap.set(5, 5)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

"""
0. CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
1. CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
3. CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file
4. CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
5. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
6. CV_CAP_PROP_FPS Frame rate.

1920 1080
1504 832
1280 720
1024 768
960 720
640 480
"""

############## Running the tensorflow session ##############

with detection_graph.as_default():
  # tf.Session (A class for running TensorFlow operations)
  with tf.Session(graph=detection_graph) as sess: # open default graph = our pre-train model "forzen model"
  
   ret = True
   while (ret):
      ret,image_np = cap.read()
	  
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
	  
	  ###
	  ### .get_tensor_by_name = > Returns the Tensor with the given name
	  ### This method may be called concurrently from multiple threads. (parallel algorithm)
	  ###
	  
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
	  
	
      #### Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
		  
		  
      ### Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=6)
		  
		  
      #out.write(image_np)
      cv2.imshow('image',cv2.resize(image_np,(1024,768)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          cap.release()
          out.release()
          break

