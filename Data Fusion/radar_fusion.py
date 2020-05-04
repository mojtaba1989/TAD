import numpy as np
import os
import sys
import time
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import python_data_fusion as pdf
import tensorflow as tf

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
    print(PATH_TO_CSV)
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'label_map.pbtxt')
    MODEL_NAME = 'inference_graph'
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

    for idx, arg in enumerate(sys.argv):
        if arg in ['--index', '-id']:
            INDEX = int(sys.argv[idx + 1])
            del sys.argv[idx]
            del sys.argv[idx]
        else:
            is_error = True
    IMG_NAME = "img_%d.jpeg" % INDEX
    PATH_TO_IMAGE = os.path.join(CWD_PATH, FILE_NAME, 'Figures/', IMG_NAME)


    for idx, arg in enumerate(sys.argv):
        if arg in ['--threshold', '-t']:
            threshold = float(sys.argv[idx + 1])
            del sys.argv[idx]
            del sys.argv[idx]


    for idx, arg in enumerate(sys.argv):
        if arg in ['--graph', '-g']:
            OUTPUT = 'graph'
            del sys.argv[idx]
        elif arg in ['--image', '-i']:
            OUTPUT = 'image'
            del sys.argv[idx]
        else:
            is_error = True

    if len(sys.argv) != 1:
        is_error = True
    else:
        for arg in sys.argv:
            if arg.startswith('-'):
                is_error = True

    if is_error:
        print('File not found')

    # Number of classes the object detector can identify
    NUM_CLASSES = 3

    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)


    # Importing the RADAR data from CSV



    index, time_stamp, angle, x_p, y_p, range_p, peakVal_p, x_d, y_d, range_d, peakVal_d, p_p, p_d = pdf.readCSV(
        PATH_TO_CSV, INDEX)



    # Importing the Image


    # image = pdf.undistort_org(PATH_TO_IMAGE)
    # image = pdf.undistort(PATH_TO_IMAGE, balance=1, dim2=None, dim3=None)

    image = cv2.imread(PATH_TO_IMAGE)
    # cv2.imshow('Object detector', cv2.resize(image, (800, 600)))
    # cv2.waitKey(0)

    # img_undistorted = pdf.undistort(image, balance=1, dim2=None, dim3=None)
    # cv2.imshow('Object detector', cv2.resize(img_undistorted, (800, 600)))
    # cv2.waitKey(0)

    # RADAR DATA fusion

    p_p = pdf.tm_f(p_p, .16, .85, .05, 30, 7, 'p')
    p_d = pdf.tm_f(p_d, .16, .85, .05, 24, 7, 'd')

    # p_p = pdf.tm_f(p_p, .0, .85, .05, 27, 0, 'p')
    # p_d = pdf.tm_f(p_d, .0, .85, .05, 19, 0, 'd')


    points = np.concatenate((p_p, p_d), axis=1)


    # Map RADAR data to Image


    keypoints = pdf.radar2image_trans_keypoint(points, .7, .06, -7, 2.1, image.shape[:-1])
    boxes = pdf.radar2image_trans(points, .7, .06, -7, 2.1, image.shape[:-1])

    classes_p = np.zeros([1, p_p.shape[1]])
    classes_d = np.zeros([1, p_d.shape[1]])
    classes_p[0, ] = 2
    classes_d[0, ] = 1
    score_p = peakVal_p/np.max(peakVal_p)
    score_d = peakVal_d/np.max(peakVal_d)

    classes = np.concatenate((classes_p, classes_d), axis=1)
    scores = np.concatenate((score_p, score_d), axis=0)





    # Lamp detection

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)


    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0') # Input tensor is the image
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')    # Output tensors are the detection
    # boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')



    image_expanded = np.expand_dims(image, axis=0)



    (boxes_ml, scores_ml, classes_ml, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    boxes_ml = boxes_ml.reshape(-1, 4)
    scores_ml = scores_ml.reshape(-1, 1)
    classes_ml = classes_ml.reshape(-1, 1)



    classes_ml.fill(3)

    # boxes_ml = pdf.undistortPoint(boxes_ml, image.shape[:-1])
    # mid = pdf.undistortPoint(mid, image.shape[:-1])



    if threshold != False:
        classes = classes[:, scores > threshold]
        keypoints = keypoints[scores > threshold, ]
        boxes = boxes[scores > threshold, ]
        scores = scores[scores > threshold]


        boxes_ml = np.array(boxes_ml)
        scores_ml = np.array(scores_ml)
        classes_ml = np.array(classes_ml)

        fii = scores_ml > threshold
        fii = fii.reshape(-1).tolist()

        classes_ml = classes_ml[fii, :]
        boxes_ml = boxes_ml[fii, :]
        scores_ml = scores_ml[fii, :]



if OUTPUT == 'graph':

    cv2.destroyAllWindows()
    mid = pdf.interp(boxes_ml)
    boxes_ml = pdf.undistortPoint(boxes_ml, image.shape[:-1])
    mid = pdf.undistortPoint(mid, image.shape[:-1])
    cloud_points = pdf.img2radar_map(mid, .8, -7, 2.1, image.shape[:-1])-np.array([[0], [.32], [0]])

    # alpha, midpoint = pdf.cam_ang_det(cloud_points, [0, .45])



    # START QtAPPfor the plot
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
    # s_d = p.plot([], [], pen=None, symbol='o', name='Driver Side')
    # s_p = p.plot([], [], pen=None, symbol='x', name='Passenger Side')
    s_cloud = p.plot([], [], pen=None, symbol='s', color='red', name='detection')
    # line = p.plot([-10, 0, 10], [4.663, 0, 4.663])
    # line_det = p.plot([0, midpoint[0]], [.29, midpoint[1]])
    # p.plot([], [], pen=None, symbol=None, name=-np.array(angle))
    # p.plot([], [], pen=None, symbol=None, name=alpha)
    # s_d.setData(p_d[0:-1, ].T)
    # s_p.setData(p_p[0:-1, ].T)
    s_cloud.setData(cloud_points[0:-1, ].T)

    while True:
        try:
            QtGui.QApplication.processEvents()
            time.sleep(.1)
        except KeyboardInterrupt:

            win.close()
            break

elif OUTPUT == 'image':

    # rows, cols, ch = image.shape

    # M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 0, 1)
    # image = cv2.warpAffine(image, M, (cols, rows))
    # image = pdf.undistort_org(image)

    # vis_util.visualize_boxes_and_labels_on_image_array(image,
    #                                                 np.squeeze(boxes_ml),
    #                                                 np.squeeze(classes_ml).astype(np.int32),
    #                                                 np.squeeze(scores_ml),
    #                                                 category_index,
    #                                                 use_normalized_coordinates=True,
    #                                                 line_thickness=8,
    #                                                 min_score_thresh=threshold)

    # cv2.imshow('Object detector', cv2.resize(image, (800, 600)))

    # Press any key to close the image
    # cv2.waitKey(0)

    mid = pdf.interp(boxes_ml)


    boxes_ml = pdf.undistortPoint(boxes_ml, image.shape[:-1])
    mid = pdf.undistortPoint(mid, image.shape[:-1])

    image = pdf.undistort_org(image)

    vis_util.draw_keypoints_on_image_array(image,
                                           mid,
                                           color='blue',
                                           radius=5,
                                           use_normalized_coordinates=True)

    # All the results have been drawn on image. Now display the image.



    image = cv2.resize(image, (800, 600))
    cv2.imshow('Object detector', image)

    # Press any key to close the image
    cv2.waitKey(0)




    # Clean up
    cv2.destroyAllWindows()