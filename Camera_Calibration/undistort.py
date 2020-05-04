import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import os

# You should replace these 3 lines with the output in calibration step
DIM = (1280, 960)
K = np.array([[585.8322184022027, 0.0, 620.377742120009], [0.0, 586.8191299992371, 480.06144414295363], [0.0, 0.0, 1.0]])
D = np.array([[-0.03134471944425443], [0.015500584682024183], [0.0055647655811822865], [-0.015794429286452086]])
def undistort(img_path):
    img = cv2.imread(img_path)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return  undistorted_img

if __name__ == '__main__':
    IMG_PATH = os.path.join(os.getcwd(), "Image__2019-09-27__14-31-39.bmp")
    image = undistort(IMG_PATH)
    cv2.imshow('Distorted', cv2.resize(cv2.imread(IMG_PATH),(800, 600)))
    cv2.waitKey(0)
    cv2.imshow('Undistorted', cv2.resize(image, (800, 600)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()