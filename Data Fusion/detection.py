import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import python_data_fusion as pdf
import time

sys.path.append("..")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util



MODEL_NAME = 'inference_graph_FRCNN'
# FILE_NAME = "04-Feb-2020-12-40"
FILE_NAME = '01-Oct-2019-13-56'
# FILE_NAME = '20-Nov-2019-11-16'

CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'label_map.pbtxt')

INDEX = 10
IMG_NAME = "img_%d.jpeg" % INDEX
PATH_TO_IMAGE = os.path.join(CWD_PATH, FILE_NAME, 'Figures/', IMG_NAME)

trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker

NUM_CLASSES = 1

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)



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

tstart = time.time()

image = cv2.imread(PATH_TO_IMAGE)
image_expanded = np.expand_dims(image, axis=0)

(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

print(time.time()-tstart)

boxes_ml = boxes.reshape(-1, 4)
boxes_ml = boxes_ml[(scores > .8).reshape(-1), ]
boxes_ml = pdf.de_normalize(boxes_ml, image.shape[0:2])

# vis_util.visualize_boxes_and_labels_on_image_array(
#                                                     image,
#                                                     np.squeeze(boxes_ml),
#                                                     np.squeeze(classes).astype(np.int32),
#                                                     np.squeeze(scores),
#                                                     category_index,
#                                                     use_normalized_coordinates=False,
#                                                     line_thickness=8,
#                                                     min_score_thresh=0.01)
#
# cv2.imshow('Object detector', cv2.resize(image, (800, 600)))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

box_temp = boxes_ml.copy()


bboxes = []

for row in range(boxes_ml.shape[0]):
    box_temp[row, 0] = boxes_ml[row, 1]
    box_temp[row, 1] = boxes_ml[row, 0]
    box_temp[row, 2] = boxes_ml[row, 3] - boxes_ml[row, 1]
    box_temp[row, 3] = boxes_ml[row, 2] - boxes_ml[row, 0]
    bbox = tuple(box_temp[row, :])
    bboxes.append(bbox)

## Select boxes
# bboxes = []
# colors = []

# OpenCV's selectROI function doesn't work for selecting multiple objects in Python
# So we will call this function in a loop till we are done selecting all objects
# while True:
#     # draw bounding boxes over objects
#     # selectROI's default behaviour is to draw box starting from the center
#     # when fromCenter is set to false, you can draw box starting from top left corner
#     bbox = cv2.selectROI('MultiTracker', cv2.resize(image, (800, 600)))
#     bboxes.append(bbox)
#     colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
#     print("Press q to quit selecting boxes and start tracking")
#     print("Press any other key to select next object")
#     k = cv2.waitKey(0) & 0xFF
#     if (k == 113):  # q is pressed
#         break
#
# print('Selected bounding boxes {}'.format(bboxes))





# Defining the tracker
# tracker  = cv2.TrackerCSRT_create()
trackerType = trackerTypes[4]
multitracker = cv2.MultiTracker_create()


for bbox in bboxes:
    multitracker.add(createTrackerByName(trackerType), image, bbox)


for INDEX in range(300):
    IMG_NAME = "img_%d.jpeg" % INDEX
    PATH_TO_IMAGE = os.path.join(CWD_PATH, FILE_NAME, 'Figures/', IMG_NAME)
    image = cv2.imread(PATH_TO_IMAGE)

    tstart = time.time()
    try:
        success, boxes = multitracker.update(image)
        print('time', INDEX, time.time()-tstart)

        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(image, p1, p2, (100, 100, 50), 2, 1)

        cv2.imshow('Multitracker', cv2.resize(image, (800, 600)))
        cv2.waitKey(1)
    except:
        pass











# mid = pdf.interp(boxes_ml)
# mid = pdf.undistortPoint(mid, image.shape[:-1])
#
#
#
# image = pdf.undistort_org(image)

# vis_util.draw_keypoints_on_image_array(image,
#                                        mid,
#                                        color='blue',
#                                        radius=5,
#                                        use_normalized_coordinates=True)
# image = cv2.resize(image, (800, 600))
# cv2.imshow('Object detector', image)
#
# # Press any key to close the image
# cv2.waitKey(0)



cv2.destroyAllWindows()