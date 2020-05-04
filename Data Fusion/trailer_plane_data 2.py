import numpy as np
import os
import sys
import glob
import cv2
import csv
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import python_data_fusion as pdf
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import tensorflow as tf
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append("..")

if __name__ == '__main__':

    threshold = False

    for idx, arg in enumerate(sys.argv):
        if arg in ['--file', '-f']:
            FILE_NAME = str(sys.argv[idx+1])
            del sys.argv[idx]
            del sys.argv[idx]
        else:
            is_error = True

    CWD_PATH = os.getcwd()
    PATH_TO_CSV = os.path.join(CWD_PATH, FILE_NAME, FILE_NAME+'.csv')
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'label_map.pbtxt')
    MODEL_NAME = 'inference_graph'
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

    for idx, arg in enumerate(sys.argv):
        if arg in ['--threshold', '-t']:
            threshold = float(sys.argv[idx + 1])
            del sys.argv[idx]
            del sys.argv[idx]

    if len(sys.argv) != 1:
        is_error = True
    else:
        for arg in sys.argv:
            if arg.startswith('-'):
                is_error = True

    if is_error:
        print('File not found')

    num = len(glob.glob(os.path.join(CWD_PATH, FILE_NAME, 'Figures/*.jpeg')))
    myFile = open(os.path.join(CWD_PATH, FILE_NAME, 'RADAR-lined-' + FILE_NAME + '.csv'), 'w', newline='')
    writer = csv.writer(myFile)
    writer.writerow(['Index', 'Processing Time', 'vernier', 'RADAR X Passenger', 'RADAR Y Passenger',
                     'RADAR Val Passenger', 'RADAR X Driver',
                     'RADAR Y Driver', 'RADAR Val Driver', 'Camera X', 'Camera Y'])

    NUM_CLASSES = 3
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Lamp detection

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    if True:
        app = QtGui.QApplication([])

        # Set the plot
        pg.setConfigOption('background', 'w')
        win = pg.GraphicsWindow(title="2D scatter plot")
        p = win.addPlot()
        p.setXRange(-2, 2)
        p.setYRange(0, 2)
        p.setLabel('left', text='Y position (m)')
        p.setLabel('bottom', text='X position (m)')
        p.addLegend(size=None)
        s_d = p.plot([], [], pen=None, symbol='o', name='Driver Side')
        s_p = p.plot([], [], pen=None, symbol='x', name='Passenger Side')
        s_cloud = p.plot([], [], pen=None, symbol='s', color='red', name='detection')




    # Main Loop

    t_old = time.time()
    
    for INDEX in range(num):
        print(INDEX)
        IMG_NAME = "img_%d.jpeg" % INDEX
        PATH_TO_IMAGE = os.path.join(CWD_PATH, FILE_NAME, 'Figures/', IMG_NAME)
        image = cv2.imread(PATH_TO_IMAGE)
        image_expanded = np.expand_dims(image, axis=0)
        (boxes_ml, scores_ml, classes_ml, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
        boxes_ml = boxes_ml.reshape(-1, 4)
        boxes_ml = boxes_ml[(scores_ml > .1).reshape(-1), ]

        mid = pdf.interp(boxes_ml)
        mid = pdf.undistortPoint(mid, image.shape[:-1])
        #
        # image = pdf.undistort_org(image)
        #
        # vis_util.draw_keypoints_on_image_array(image,
        #                                        mid,
        #                                        color='blue',
        #                                        radius=5,
        #                                        use_normalized_coordinates=True)
        # image = cv2.resize(image, (800, 600))
        # cv2.imshow('Object detector', image)
        #
        # # Press any key to close the image
        # cv2.waitKey(1)

        cloud_points = pdf.img2radar_map(mid, .78, -7, 2.1, image.shape[:-1])

        index, time_stamp, angle, x_p, y_p, range_p, peakVal_p, x_d, y_d, range_d, peakVal_d, p_p, p_d = pdf.readCSV(
            PATH_TO_CSV, INDEX)
        p_p = pdf.tm_f(p_p, .16, .85, .05, 30, 7, 'p')
        p_d = pdf.tm_f(p_d, .16, .85, .05, 24, 7, 'd')

        s_d.setData(p_d[0:-1, ].T)
        s_p.setData(p_p[0:-1, ].T)
        s_cloud.setData(cloud_points[0:-1, ].T)
        QtGui.QApplication.processEvents()


        writer.writerow([INDEX, t_old - time.time(), angle, p_p[0, ], p_p[1, ], peakVal_p, p_d[0, ], p_d[1, ],
                         peakVal_d,
                         cloud_points[0, ],
                         cloud_points[1, ]])
        t_old = time.time()
