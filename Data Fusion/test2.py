# import numpy as np
# import os
# import sys
# import time
# import pyqtgraph as pg
# from pyqtgraph.Qt import QtGui
# import cv2
# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as vis_util
# import python_data_fusion as pdf
# import tensorflow as tf
#
# CWD_PATH = '/Users/moj/TAD/Data Fusion'
# PATH_TO_LABELS = os.path.join(CWD_PATH, 'label_map.pbtxt')
# MODEL_NAME = 'inference_graph'
# PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
# IMG_NAME = 'img_10.jpeg'
# PATH_TO_IMAGE = os.path.join(CWD_PATH, IMG_NAME)
#
# NUM_CLASSES = 3
# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(label_map,
#                                                             max_num_classes=NUM_CLASSES,
#                                                             use_display_name=True)
# category_index = label_map_util.create_category_index(categories)
#
# image = cv2.imread(PATH_TO_IMAGE)
#
#
# detection_graph = tf.Graph()
# with detection_graph.as_default():
#     od_graph_def = tf.compat.v1.GraphDef()
#     with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
#         serialized_graph = fid.read()
#         od_graph_def.ParseFromString(serialized_graph)
#         tf.import_graph_def(od_graph_def, name='')
# sess = tf.Session(graph=detection_graph)
#
# image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')  # Input tensor is the image
# detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')  # Output tensors are the detection
# detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
# detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
# num_detections = detection_graph.get_tensor_by_name('num_detections:0')
#
# image_expanded = np.expand_dims(image, axis=0)
#
# (boxes_ml, scores_ml, classes_ml, num) = sess.run(
#     [detection_boxes, detection_scores, detection_classes, num_detections],
#     feed_dict={image_tensor: image_expanded})
#
# classes_ml.fill(3)
# boxes_ml = boxes_ml[:, 0:int(num), :]
# print('before', boxes_ml)
# boxes_ml = pdf.undistortPoint(boxes_ml, image.shape[:-1])
# print('after', boxes_ml)
#
#
#
# imgg = np.zeros([960, 1280])
#
# # imgg[400, 400] = 255
#
# # (array([395]), array([386]))
#
#
#
# cv2.imshow('a', undistorted_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# myFile = open(os.path.join(CWD_PATH, FILE_NAME, 'TPDA' + FILE_NAME + '.csv'), 'w', newline='')
# writer = csv.writer(myFile)
# writer.writerow(['Index', 'vernier', 'Camera 1', 'Camera 2'])
#
# INDEX = 0
# while True:
#     vernier, x_p, y_p, x_d, y_d, x_c, y_c, p_p, p_d, p_c = pdf.readTPCSV(PATH_TO_CSV, INDEX)
#     out = pdf.cam_ang_det_L(p_c)
#     writer.writerow([INDEX, vernier, out[0], out[1]])
#     INDEX += 1
#
#     writer.writerow([idx, np.array(vernier).item(0), np.array(x_c).item(0), np.array(y_c).item(0),
#                      np.array(x_c).item(1), np.array(y_c).item(1)])


# from __future__ import absolute_import, division, print_function, unicode_literals
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
import python_data_fusion as pdf
import os
import cv2



# PATH_TO_MODEL = 'Angle_Detection_Model/angle_detection_estimator.h5'
# model = tf.keras.models.load_model(PATH_TO_MODEL, custom_objects={'TAD_loss': pdf.TAD_loss})
#
# PATH_TO_CSV = os.path.join(os.getcwd(), '01-Oct-2019-14-04/TPDA-lined-cleaned-01-Oct-2019-14-04.csv')
#
# data = pd.read_csv(PATH_TO_CSV,
#                       na_values = "?", comment='\t',
#                       sep=",", skipinitialspace=True)
#
# dataset = data.copy()
#
# dataset = pdf.norm(dataset[["T_est", "d_T_est", "T_est_inv"]])
# # dataset = dataset.dropna()
#
# ML_est = model.predict(pdf.norm(dataset))
#
# data.insert(63, "ML_est_new", ML_est.reshape(-1).tolist(), True)
# data["New_Ver"] = data["New_Ver"].shift(-20)
# data = data.dropna()
#
# data.to_csv(os.path.join(os.getcwd(), 'Final.csv'), index=None, header=True)

# FILE_NAME = '01-Oct-2019-13-56'
# CWD_PATH = os.getcwd()
# Data = pd.read_csv(os.path.join(CWD_PATH, FILE_NAME, 'RADAR-cleaned-' + FILE_NAME + '.csv'),
#                       na_values="?", comment='\t',
#                       sep=",", skipinitialspace=True)
# Data["New_Ver"] = Data["New_Ver"].shift(-20)
# Data["Vernier"] = Data["Vernier"].shift(-20)
# Data = Data.dropna()
# Data.to_csv(os.path.join(CWD_PATH, FILE_NAME, 'RADAR-cleaned-' + FILE_NAME + '.csv'))


# FILE_NAME = '01-Oct-2019-14-04'
# FILE_NAME = '20-Nov-2019-11-06'
FILE_NAME = '04-Feb-2020-12-45'
CWD_PATH = os.getcwd()
PATH_TO_IMAGE = os.path.join(CWD_PATH, FILE_NAME, 'Figures', 'img_68.jpeg')
# PATH_TO_CSV = os.path.join(CWD_PATH, FILE_NAME, 'RADAR-cleaned-'+FILE_NAME+'.csv')

image = cv2.imread(PATH_TO_IMAGE)
# image = pdf.img_filter(image, v_lim=[.5, 1])
# image = cv2.resize(image, (800, 600))
cv2.imshow('Object detector', cv2.resize(image, (800, 600)))
# # Press any key to close the image
cv2.waitKey(0)
# print()
image = pdf.undistort_org(image)
cv2.imshow('Object detector', cv2.resize(image, (800, 600)))
# # Press any key to close the image
cv2.waitKey(0)

# Data =  pd.read_csv(PATH_TO_CSV,
#                       na_values="?", comment='\t',
#                       sep=",", skipinitialspace=True)



# g = sns.pairplot(Data[["Cam", "Cam+Hitch"]], diag_kind="kde")


# g = sns.pairplot(Data[["Cam", "Cam+Hitch", "Radar",
#            "Radar+Hitch", "Radar+Cam", 'Radar+Cam+Hitch']], diag_kind="kde")

# g = sns.pairplot(Data[["Cam+Hitch",
#            "Radar+Hitch", 'Radar+Cam+Hitch']], diag_kind="kde")

# g = sns.pairplot(Data[["Cam",
#            "Radar", 'Radar+Cam']], diag_kind="kde")

# g = sns.pairplot(Data[["Radar", "Cam"]], diag_kind="kde")

# plt.savefig(FILE_NAME+'/pairplot9.png')
#
