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

if __name__ == '__main__':

    IMG_NAME = "img_10.jpeg"
    CWD_PATH = os.getcwd()
    PATH_TO_IMAGE = os.path.join(CWD_PATH, '01-Oct-2019-13-56', 'Figures', IMG_NAME)



    PATH_TO_LABELS = os.path.join(CWD_PATH, 'label_map.pbtxt')


    for idx, arg in enumerate(sys.argv):
        if arg in ['-x']:
            x = float(sys.argv[idx + 1])
            del sys.argv[idx]
            del sys.argv[idx]
        else:
            is_error = True

    for idx, arg in enumerate(sys.argv):
        if arg in ['-y']:
            y = int(sys.argv[idx + 1])
            del sys.argv[idx]
            del sys.argv[idx]
        else:
            is_error = True

    for idx, arg in enumerate(sys.argv):
        if arg in ['-z']:
            z = int(sys.argv[idx + 1])
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

    if len(sys.argv) != 1:
        is_error = True
    else:
        for arg in sys.argv:
            if arg.startswith('-'):
                is_error = True

    if is_error:
        print('File not found')


    # Number of classes the object detector can identify
    NUM_CLASSES = 2

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


    # points = np.vstack((x, y))

    points = np.zeros([3, 48])
    k = 0
    for x in np.arange(-.6, 1, .6):
        for y in np.arange(0, 1.6, .1):
            points[0, k] = x
            points[1, k] = y
            k = k + 1

    # point_p = np.array([[-.79],
    #                     [1.49]])
    # point_p = pdf.tm_f(point_p, .4, .79, 5e-2, 25, 10, 'd')
    #
    # point_d = np.array([[.99],
    #                     [1.46]])
    # point_d = pdf.tm_f(point_d, .4, .79, 5e-2, 25, 10, 'p')
    # points = np.concatenate((point_d, point_p), axis=1)
    # print(points)
    # keypoints = pdf.radar2image_trans_keypoint(points, .7, 0, 13, 2.1, [4.8, 3.6])
    keypoints = pdf.radar2image_trans_keypoint(points, .7, .06, -7, 2.1, [4.8, 3.6])




    if OUTPUT == 'graph':
        # START QtAPPfor the plot
        app = QtGui.QApplication([])

        # Set the plot
        pg.setConfigOption('background', 'w')
        win = pg.GraphicsWindow(title="2D scatter plot")
        pg.LegendItem()
        p = win.addPlot()
        p.setXRange(-10, 10)
        p.setYRange(0, 10)
        p.setLabel('left', text='Y position (m)')
        p.setLabel('bottom', text='X position (m)')
        p.addLegend(size= None)
        s = p.plot([], [], pen=None, symbol='o',name = 'Driver Side')
        line = p.plot([-10, 0, 10], [4.663, 0, 4.663])
        s.setData(points[:-1].T)

        #
        # print(angle)
        #
        while True:
            try:
                QtGui.QApplication.processEvents()
                time.sleep(.1)
            except KeyboardInterrupt:

                win.close()
                break

    elif OUTPUT == 'image':
        image = pdf.undistort(PATH_TO_IMAGE)
        rows, cols, ch = image.shape

        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 0, 1)
        image = cv2.warpAffine(image, M, (cols, rows))

        vis_util.draw_keypoints_on_image_array(image,
                            keypoints,
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

