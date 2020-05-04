import glob
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import pyqtgraph as pg
import python_data_fusion as pdf
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from pyqtgraph.Qt import QtGui

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append("..")

if __name__ == '__main__':
    for idx, arg in enumerate(sys.argv):
        if arg in ['--file', '-f']:
            FILE_NAME = str(sys.argv[idx + 1])
            del sys.argv[idx]
            del sys.argv[idx]
        else:
            is_error = True

    for idx, arg in enumerate(sys.argv):
        if arg in ['--graph', '-g']:
            OUTPUT = 'graph'
            del sys.argv[idx]
        elif arg in ['--image', '-i']:
            OUTPUT = 'image'
            del sys.argv[idx]
        else:
            is_error = True

    CWD_PATH = os.getcwd()
    PATH_TO_CSV = os.path.join(CWD_PATH, FILE_NAME, FILE_NAME + '.csv')
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'label_map.pbtxt')
    MODEL_NAME = 'inference_graph_FRCNN'
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

    # making output csv
    num = len(glob.glob(os.path.join(CWD_PATH, FILE_NAME, 'Figures/*.jpeg')))
    PATH_TO_RESULTS = os.path.join(CWD_PATH, FILE_NAME, 'RADAR-Result-' + FILE_NAME + '.csv')
    Results = pd.DataFrame(np.zeros([num, 1]), columns=["Vernier"])

    for idx, arg in enumerate(sys.argv):
        if arg in ['--threshold', '-t']:
            threshold = float(sys.argv[idx + 1])
            del sys.argv[idx]
            del sys.argv[idx]
        else:
            threshold = False

    if len(sys.argv) != 1:
        is_error = True
    else:
        for arg in sys.argv:
            if arg.startswith('-'):
                is_error = True

    # MODEL_NAME = 'inference_graph_FRCNN'
    # # FILE_NAME = "04-Feb-2020-12-40"
    # FILE_NAME = '01-Oct-2019-13-56'
    # # FILE_NAME = '20-Nov-2019-11-16'
    #
    # CWD_PATH = os.getcwd()
    # PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
    # PATH_TO_LABELS = os.path.join(CWD_PATH, 'label_map.pbtxt')
    # PATH_TO_CSV = os.path.join(CWD_PATH, FILE_NAME, FILE_NAME+'.csv')
    # threshold = .8
    # num = len(glob.glob(os.path.join(CWD_PATH, FILE_NAME, 'Figures/*.jpeg')))
    # INDEX = 10
    # IMG_NAME = "img_%d.jpeg" % INDEX
    # PATH_TO_IMAGE = os.path.join(CWD_PATH, FILE_NAME, 'Figures/', IMG_NAME)
    # columns = ['Vernier', "Cam", "Radar", "L", "Lambda", "Hitch-Angle", "e(Cam)", "e(Radar)", "e(Hitch-Angle)"]
    # Results = pd.DataFrame(np.zeros([num, 9]), columns=columns)

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

    # Lamp tracking
    TRACK_FLAG = False
    multitracker = cv2.MultiTracker_create()

    # Adjust the center
    # center = np.array([[-.1], [.28]])
    center = np.array([[-.0872], [.1282]])
    # Initialization
    p_r_old = np.nan
    p_val_old = np.nan
    d_r_old = np.nan
    d_val_old = np.nan
    #
    w = [1, 1, 10]  # W = [W_L, W_V, W_C]
    #
    rdata = np.ones([10]) * 10
    cdata = np.ones([10]) * 10

    cam_l_corner = np.nan
    cam_w_trailer = np.nan
    cam_var = 1

    l_p = np.nan
    l_d = np.nan
    #
    radar_l_corner = np.nan
    radar_w_trailer = np.nan
    radar_var = 1

    var_t = .01

    if OUTPUT == 'graph':
        app = QtGui.QApplication([])

        # Set the plot
        pg.setConfigOption('background', 'w')
        win = pg.GraphicsWindow(title="2D scatter plot")
        p = win.addPlot()
        p.setXRange(-3, 3)
        p.setYRange(0, 3)
        p.setLabel('left', text='Y position (m)')
        p.setLabel('bottom', text='X position (m)')
        p.addLegend(size=None)
        s_d = p.plot([], [], pen=None, symbol='o', name='Driver Side')
        s_p = p.plot([], [], pen=None, symbol='x', name='Passenger Side')
        s_cloud = p.plot([], [], pen=None, symbol='s', color='red', name='radar')

    # Main Loop
    for INDEX in range(num):
        tstart = time.time()
        index, time_stamp, angle, x_p, y_p, range_p, peakVal_p, x_d, y_d, range_d, peakVal_d, p_p, p_d = pdf.readCSV(
            PATH_TO_CSV, INDEX)
        Results.loc[INDEX, "Vernier"] = np.array(angle)
        p_p = pdf.tm_f(p_p, .16, .85, .05, 25, 6, 'p')[0:2, ] - center
        p_d = pdf.tm_f(p_d, .16, .85, .05, 20, 6, 'd')[0:2, ] - center
        val_p = peakVal_p
        val_d = peakVal_d

        IMG_NAME = "img_%d.jpeg" % index
        PATH_TO_IMAGE = os.path.join(CWD_PATH, FILE_NAME, 'Figures/', IMG_NAME)
        image = cv2.imread(PATH_TO_IMAGE)

        if not TRACK_FLAG:
            image_expanded = np.expand_dims(image, axis=0)
            (boxes_ml, scores_ml, classes_ml, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_expanded})
            boxes_ml = boxes_ml.reshape(-1, 4)
            if threshold is not False:
                boxes_ml = boxes_ml[(scores_ml > threshold).reshape(-1),]
            boxes_ml = pdf.tfcv_convertor(boxes_ml, image.shape[0:2], source='tf')
            for bbox in boxes_ml:
                multitracker.add(cv2.TrackerMedianFlow_create(), image, bbox)
            TRACK_FLAG = True
            boxes_ml = pdf.tfcv_convertor(boxes_ml, image.shape[0:2], source='cv')

        else:
            success, boxes_ml = multitracker.update(image)
            boxes_ml = pdf.tfcv_convertor(boxes_ml, image.shape[0:2], source='cv')

        mid = pdf.undistortPoint(pdf.mid_detection(boxes_ml), image.shape[:-1])
        mid[mid == 0] = np.nan
        mid = mid[~np.isnan(mid).any(axis=1)]

        cam_det = pdf.img2radar_map(mid, .78, -6, 2.1, image.shape[:-1])

        # Image processing:
        x_c = cam_det[0,]
        y_c = cam_det[1,]
        y_c = y_c[x_c.argsort()]
        x_c.sort()

        if len(x_c) == 2:
            missed_flag = False
            p_c = np.array([[x_c[0]], [y_c[0]]]) - center
            d_c = np.array([[x_c[1]], [y_c[1]]]) - center

            if abs(p_c[0]) >= abs(d_c[0]):
                missed_marker_light = 'p'
            else:
                missed_marker_light = 'd'

        elif len(x_c) == 1:
            missed_flag = True
            if missed_marker_light == 'p':
                d_c = np.array([[x_c[0]], [y_c[0]]]) - center
                p_c = np.array([[np.nan], [np.nan]])
            else:
                p_c = np.array([[x_c[0]], [y_c[0]]]) - center
                d_c = np.array([[np.nan], [np.nan]])

        R = np.concatenate([p_c, d_c], axis=1)

        cam_phi, cam_l_corner, cam_w_trailer = pdf.update_measure(
            R, cam_l_corner, cam_w_trailer, cam_var, var_t, missed_flag, missed_marker_light)

        cam_var, cdata = pdf.res_var(cdata, cam_l_corner, n=10)

        Results.loc[INDEX, "Cam"] = cam_phi
        Results.loc[INDEX, "e(Cam)"] = Results.loc[INDEX, "Vernier"] - cam_phi

        # Point-Cloud Processing:
        p_r, p_val = pdf.Cam_Radar(
            p_p, val_p, p_r_old, p_val_old, p_c, w, method='kalmann')
        d_r, d_val = pdf.Cam_Radar(
            p_d, val_d, d_r_old, d_val_old, d_c, w, method='kalmann')

        R = np.concatenate([p_r, d_r], axis=1)

        radar_phi, radar_l_corner, radar_w_trailer = pdf.update_measure(
            R, radar_l_corner, radar_w_trailer, radar_var, var_t, False, missed_marker_light)

        radar_var, rdata = pdf.res_var(rdata, radar_phi, n=10)

        Results.loc[INDEX, "Radar"] = radar_phi
        Results.loc[INDEX, "e(Radar)"] = Results.loc[INDEX, "Vernier"] - radar_phi
        Results.loc[INDEX, "L"] = cam_l_corner
        Results.loc[INDEX, "Lambda"] = cam_w_trailer

        p_r_old = p_r
        p_val_old = p_val
        d_r_old = d_r
        d_val_old = d_val

        tend = time.time()
        Results.loc[INDEX, "Time"] = time.time() - tstart

        # pose = np.array([[0, 0, w_trailer / 2, -w_trailer / 2], [0, l_trailer, l_trailer, l_trailer]])
        # phi = np.radians(phi)
        # c, s = np.cos(phi), np.sin(phi)
        # TF = np.array(((c, s), (-s, c))).reshape(2, 2)
        # pose = TF.dot(pose)

        if OUTPUT == 'image':
            image = pdf.undistort_org(image)
            text = 'FPS:%d' % int(1 / Results.loc[INDEX, "Time"])
            image = cv2.putText(image, text, (1100, 900), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 2, cv2.LINE_AA)

            vis_util.draw_keypoints_on_image_array(image,
                                                   mid,
                                                   color='blue',
                                                   radius=5,
                                                   use_normalized_coordinates=True)
            cv2.imshow('Object detector', cv2.resize(image, (800, 600)))
            cv2.waitKey(1)

        if OUTPUT == 'graph':
            s_d.setData(p_c[0:-1, ].T)
            s_p.setData(d_c[0:-1, ].T)
            s_cloud.setData(R[0:-1, ].T)
            QtGui.QApplication.processEvents()

    # cv2.destroyAllWindows()

    Results = Results.dropna()
    Results.to_csv(PATH_TO_RESULTS, index=True)
    #
    # plt.ion()
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # line1, = ax1.plot(Results["Vernier"], Results["Cam"], label='Vernier')
    # # line2, = ax1.plot(Results["Cam"], label='Cam')
    # # line3, = ax1.plot(Results["Hitch-Angle"], label='Hitch-Angle')
    # # line7, = ax1.plot(Results["Radar"], label='RADAR')
    # #
    # ax1.legend()
    # plt.show()
    # plt.waitforbuttonpress()
