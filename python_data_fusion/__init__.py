from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import os
import sys
import csv
import cv2




DIM = (1280, 960)
K = np.array([[585.8322184022027, 0.0, 620.377742120009], [0.0, 586.8191299992371, 480.06144414295363],
              [0.0, 0.0, 1.0]])
D = np.array([[-0.03134471944425443], [0.015500584682024183], [0.0055647655811822865], [-0.015794429286452086]])

map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
map_inv = np.zeros([map1.shape[0], map1.shape[1], 2])


for i in range(map1.shape[0]):
    for j in range(map1.shape[1]):
        j_new = map1[i, j, 0]
        i_new = map1[i, j, 1]
        map_inv[i_new, j_new, 0] = i
        map_inv[i_new, j_new, 1] = j


def undistort(img_path, balance=0.0, dim2=None, dim3=None):

    if type(img_path) == str:
        img = cv2.imread(img_path)
    else:
        img = img_path

    dim1 = img.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
    assert dim1[0] / dim1[1] == DIM[0] / DIM[
        1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document
    # failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


def undistort_org(img_path):
    if type(img_path) == str:
        img = cv2.imread(img_path)
    else:
        img = img_path

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


def undistortPoint(pts, sensor_size):
    sensor_size = np.array(sensor_size) - 1
    pts = de_normalize(pts, sensor_size)

    if pts.shape[-1] == 2:
        for i in range(pts.shape[0]):
            ptt = pts[i, ]
            pts[i, ] = map_inv[ptt[0], ptt[1]]
    else:
        for i in range(pts.shape[0]):
            ptt = pts[i, 0:2]
            pts[i, 0:2] = map_inv[ptt[0], ptt[1]]
            ptt = pts[i, 2:4]
            pts[i, 2:4] = map_inv[ptt[0], ptt[1]]

    return normalize(pts, sensor_size)


def de_normalize(pts, sensor_size):
    if pts.shape[-1] == 2:
        pts[:, 0] = pts[:, 0] * sensor_size[0]
        pts[:, 1] = pts[:, 1] * sensor_size[1]

    else:
        pts[:, 0] = pts[:, 0] * sensor_size[0]
        pts[:, 1] = pts[:, 1] * sensor_size[1]
        pts[:, 2] = pts[:, 2] * sensor_size[0]
        pts[:, 3] = pts[:, 3] * sensor_size[1]

    return pts.astype(int)


def normalize(pts, sensor_size):
    pts = pts.astype(float)
    if pts.shape[-1] == 2:
        pts[:, 0] = pts[:, 0] / sensor_size[0]
        pts[:, 1] = pts[:, 1] / sensor_size[1]
    else:
        pts[:, 0] = pts[:, 0] / sensor_size[0]
        pts[:, 1] = pts[:, 1] / sensor_size[1]
        pts[:, 2] = pts[:, 2] / sensor_size[0]
        pts[:, 3] = pts[:, 3] / sensor_size[1]

    return pts


def get_start():
    is_error = False

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

    for idx, arg in enumerate(sys.argv):
        if arg in ['--index', '-i']:
            INDEX = int(sys.argv[idx + 1])
            del sys.argv[idx]
            del sys.argv[idx]
        else:
            is_error = True
    IMG_NAME = "img_%d.jpeg" % INDEX
    PATH_TO_FIGURE = os.path.join(CWD_PATH, FILE_NAME, 'Figures/', IMG_NAME)
    if len(sys.argv) != 1:
        is_error = True
    else:
        for arg in sys.argv:
            if arg.startswith('-'):
                is_error = True

    if is_error:
        print('File not found')
    else:
        return FILE_NAME, PATH_TO_CSV, PATH_TO_FIGURE, INDEX, PATH_TO_LABELS


def csvCellReader(init):
    init = init.replace(']', '').replace('[', '').replace('\n', '').replace(',', '')
    new = ''
    check = True
    for idx in range(len(init)):
        if init[idx] == ' ' and check:
            check = True
        elif init[idx] == ' ' and check == False:
            new += ','
            check = True
        else:
            new += init[idx]
            check = False
    new = new.split(',')
    if new[-1] == '':
        new = [float(x) for x in new[:-1]]
    else:
        new = [float(x) for x in new]
    return new


def tm_f(data, h, d1, d2, alpha, beta, side): # radar data fuison

    """
    :param data: Input raw radar data, each column is a x,y of cloud point
    :param h: radar heigh
    :param d1: distance of trunk edge to center [m]
    :param d2:  distance of antenna to the trunk edge [m]
    :param alpha: radar tilt in degree from trunk plate
    :param beta: radar tilt in degree from horizon
    :param side: 'passenger' or 'driver'
    :return: transformed matrix of the cloud points
    """
    alpha = alpha * np.pi/180
    beta = beta * np.pi/180

    Ca = np.cos(alpha)
    Sa = np.sin(alpha)

    if side == 'passenger' or side == 'p':
        T_alpha = np.array([[Ca, -Sa, 0], [Sa, Ca, 0], [0, 0, 1]])
        sign  = -1
    else:
        T_alpha = np.array([[Ca, Sa, 0], [-Sa, Ca, 0], [0, 0, 1]])
        sign  = 1


    Cb = np.cos(beta)
    Sb = np.sin(beta)

    T_beta = np.array([[1, 0, 0], [0, Cb, Sb], [0, -Sb, Cb]])

    D = np.array([[sign*(d1+Ca*d2)], [-d2*Sa], [h]])
    D = np.repeat(D, data.shape[1], axis=1)

    data_new = np.zeros([3, data.shape[1]])
    data_new[0:-1, ] = data

    data_new = T_alpha.dot(T_beta.dot(data_new))+D
    return data_new


def radar2image_trans(data, h0, h1, gamma, f, sensor_size):
    """
    :param data: Input x y cloud point detected from radar
    :param h0: height of camera from radar antenna plane
    :param h1: height of camera lens from bottom of the plate
    :param gamma: tilt of camera to ground [degree]
    :param f: focal length of camera in mm
    :param sensor_size: in mm [W,H]
    :return: pixels correspond to the cloud point as boxes
    """

    gamma = gamma * np.pi/180
    C = np.cos(gamma)
    S = np.sin(gamma)
    T_rotate = np.array([[1, 0, 0], [0, C, S], [0, -S, C]])
    T_change = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
    D = np.array([[-.01], [-h0-h1*C], [-h1*S]])

    data_new = T_rotate.dot(T_change.dot(data)+D)

    cte = 4.8 / 1280
    sensor_size = np.array(sensor_size) * cte
    O = np.array(sensor_size)/2

    u = (-f*data_new[0, ]/data_new[2, ] + O[0])/sensor_size[1]
    v = (-f*data_new[1, ]/data_new[2, ] + O[1])/sensor_size[0]


    boxes = np.zeros([data.shape[1], 4])
    boxes[:, 0] = v-.01
    boxes[:, 1] = u-.01
    boxes[:, 2] = v+.01
    boxes[:, 3] = u+.01

    return boxes


def radar2image_trans_keypoint(data, h0, h1, gamma, f, sensor_size):
    """
    :param data: Input x y cloud point detected from radar
    :param h0: height of camera from radar antenna plane
    :param h1: height of camera lens from bottom of the plate
    :param gamma: tilt of camera to ground [degree]
    :param f: focal length of camera in mm
    :param sensor_size: in mm [W,H]
    :return: pixels correspond to the cloud point as boxes
    """


    gamma = gamma * np.pi/180
    C = np.cos(gamma)
    S = np.sin(gamma)
    T_rotate = np.array([[1, 0, 0], [0, C, S], [0, -S, C]])
    T_change = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
    D = np.array([[0], [-h0 -h1*C], [-h1*S]])

    data_new = T_rotate.dot(T_change.dot(data)+D)

    cte = 4.8 / 1280
    sensor_size = np.array(sensor_size) * cte
    O = np.array(sensor_size)/2

    u = (-f*data_new[0, ]/data_new[2, ] + O[1])/sensor_size[1]
    v = (-f*data_new[1, ]/data_new[2, ] + O[0])/sensor_size[0]

    points = np.zeros([data.shape[1], 2])
    points[:, 0] = v
    points[:, 1] = u


    return points


def readCSV(PATH_TO_CSV, INDEX):
    with open(PATH_TO_CSV) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for i in range(INDEX):  # count from 0 to 7
            next(readCSV)  # and discard the rows
        row = next(readCSV)
        index = int(row[0])
        time_stamp  = float(row[1])
        angle = csvCellReader(row[2])
        x_p   = csvCellReader(row[3])
        y_p   = csvCellReader(row[4])
        range_p = csvCellReader(row[5])
        peakVal_p = csvCellReader(row[6])
        x_d   = csvCellReader(row[7])
        y_d   = csvCellReader(row[8])
        range_d = csvCellReader(row[9])
        peakVal_d = csvCellReader(row[10])

        # Meging x,y data
        p_p = np.vstack((x_p, y_p))
        p_d = np.vstack((x_d, y_d))



    return index, time_stamp, angle, x_p, y_p, range_p, peakVal_p, x_d, y_d, range_d, peakVal_d, p_p, p_d


def readTPCSV(PATH_TO_CSV, INDEX):
    with open(PATH_TO_CSV) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for i in range(INDEX):  # count from 0 to 7
            next(readCSV)  # and discard the rows
        row = next(readCSV)
        index = int(row[0])
        time_stamp  = float(row[1])
        vernier = csvCellReader(row[2])
        x_p   = csvCellReader(row[3])
        y_p   = csvCellReader(row[4])
        x_d   = csvCellReader(row[5])
        y_d   = csvCellReader(row[6])
        x_c = csvCellReader(row[7])
        y_c = csvCellReader(row[8])

        # Meging x,y data
        p_p = np.vstack((x_p, y_p))
        p_d = np.vstack((x_d, y_d))
        p_c = np.vstack((x_c, y_c))

    return index, vernier, x_p, y_p, x_d, y_d, x_c, y_c, p_p, p_d, p_c


def readValCSV(PATH_TO_CSV, INDEX):
    with open(PATH_TO_CSV) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for i in range(INDEX):  # count from 0 to 7
            next(readCSV)  # and discard the rows
        row = next(readCSV)
        index = int(row[0])
        time_stamp  = float(row[1])
        vernier = csvCellReader(row[2])
        x_p   = csvCellReader(row[3])
        y_p   = csvCellReader(row[4])
        val_p   = csvCellReader(row[5])
        x_d   = csvCellReader(row[6])
        y_d   = csvCellReader(row[7])
        val_d   = csvCellReader(row[8])
        x_c = csvCellReader(row[9])
        y_c = csvCellReader(row[10])

        # Meging x,y data
        p_p = np.vstack((x_p, y_p))
        p_d = np.vstack((x_d, y_d))
        p_c = np.vstack((x_c, y_c))

    return index, vernier, x_p, y_p, val_p, x_d, y_d, val_d, x_c, y_c, p_p, p_d, p_c


def readTPDACSV(PATH_TO_CSV, INDEX):
    with open(PATH_TO_CSV) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for i in range(INDEX):
            next(readCSV)
        row = next(readCSV)
        index = int(row[0])
        vernier = csvCellReader(row[1])
        x_c1   = csvCellReader(row[2])
        y_c1   = csvCellReader(row[3])
        x_c2   = csvCellReader(row[4])
        y_c2   = csvCellReader(row[5])

    return index, vernier, x_c1, y_c1, x_c2, y_c2


def img2radar_map(data, h, theta, f, sensor_size):
    """
    :param data: N by 2 matrix of (u,v) ratio -> note: the input is in [0,1] number type
    :param h: distance between camera and radar plane in z direction in meter
    :param theta: tilt ratio of camera (positive direction is toward the ground
    :param f: focal length
    :param sensor_size: in pixel
    :param input_type: 'box', 'point'
    :return: corresponding 3 by N matrix of cloud_point to each pixel
    """
    if data.shape[1] == 4:
        data[:, 0] = np.mean(data[:, [0, 2]], axis=1).T
        data[:, 1] = np.mean(data[:, [1, 3]], axis=1).T
        data = data[:, 0:-2]

    data = np.flip(data, axis=1)

    theta = theta*np.pi/180
    mid = np.array([[.5, .5]])


    cte = 4.8 / 1280
    sensor_size = np.array(sensor_size) * cte
    sensor_size = np.flip(sensor_size, axis=0)

    data_0 = (mid-data)/f*sensor_size
    data_0 = data_0.T
    c = np.cos(theta)
    s = np.sin(theta)
    p_0 = np.zeros([3, data_0.shape[1]])
    p_0[2, ] = h/(s-c*data_0[1, ])
    p_0[1, ] = p_0[2, ]*data_0[1, ]
    p_0[0, ] = p_0[2, ]*data_0[0, ]

    R = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
    C = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
    D = np.array([[0], [0], [h]])

    output = D + R.dot(C.dot(p_0))
    return output


def cam_ang_det(data, hitch_p):
    """
    :param data: detected boxes
    :param hitch_p: hitch ball position [x , y]
    :return: angle and midpoint of trailer
    """
    midpoint = np.mean(data, axis=1)[:-1, ]
    alpha = (midpoint[0]-hitch_p[0])/(midpoint[1]-hitch_p[1])
    angle = np.arctan(alpha)*180/np.pi

    return angle, midpoint


def cam_ang_det_L(p_c):
    l = 1.23  #trairler width and bar length - seems to be the same
    alpha = np.degrees(np.arctan(.5))
    xHat  = np.array(p_c[0, ]) / l
    beta  = np.degrees(np.arctan(xHat/np.sqrt(1.25 - xHat**2))).reshape(1, -1)
    t_a = np.sort(np.concatenate((beta - alpha, beta + alpha), axis=1))
    return t_a[:, 1:3]


def mid_detection(boxes):
    boxes = boxes.reshape(-1, 4)
    if boxes.shape[0] == 2:
        v = np.mean(boxes[:, [0, 2]], axis=1)
        u = np.mean(boxes[:, [1, 3]], axis=1)
        v_new = np.linspace(v[0], v[1], 2).reshape(-1, 1)
        u_new = np.linspace(u[0], u[1], 2).reshape(-1, 1)
        intp = np.concatenate([v_new, u_new], axis=1).reshape(-1, 2)

    else:
        v = np.mean(boxes[:, [0, 2]], axis=1)
        u = np.mean(boxes[:, [1, 3]], axis=1)
        intp = np.concatenate([v, u], axis=1).reshape(-1, 2)

    return intp


def norm(x):
    stats = x.describe().transpose()
    return (x- stats['mean']) / stats['std']


def interp(boxes):
    boxes = boxes.reshape(-1, 4)
    if boxes.shape[0] == 2:
        v = np.mean(boxes[:, [0, 2]], axis=1)
        u = np.mean(boxes[:, [1, 3]], axis=1)
        v_new = np.linspace(v[0], v[1], 20).reshape(-1, 1)
        u_new = np.linspace(u[0], u[1], 20).reshape(-1, 1)
        intp = np.concatenate([v_new, u_new], axis=1).reshape(-1, 2)

    else:
        v = np.mean(boxes[:, [0, 2]], axis=1)
        u = np.mean(boxes[:, [1, 3]], axis=1)
        intp = np.concatenate([v, u], axis=1).reshape(-1, 2)

    return intp


def TAD_loss(labels, predictions):
    import tensorflow as tf
    abs_r = tf.keras.backend.abs(labels-predictions)
    max_error = tf.keras.backend.abs(labels) / 10 + .25
    loss = tf.keras.backend.mean(tf.keras.backend.less_equal(abs_r, max_error))
    return loss


def dist(a, b):
    return np.sqrt(np.sum((a-b)**2, axis=0)).reshape(1, -1)


def distline(a, l):
    return np.abs(a[0, ]*l[0]+a[1, ]*l[1]+l[2])/np.sqrt(l[0]**2+l[1]**2).reshape(1, -1)


def Cam_Radar(p_r, val_r, p_old, val_old, p_c, w, ll=np.nan, method='kalman'):

    if method == 'kalman':
        p_r     = np.array(p_r).reshape(2, -1)
        val_r   = np.array(val_r).reshape(1, -1)
        if p_r.shape[1] > 10:
            D = np.zeros([3, 10])
        else:
            D = np.zeros([3, p_r.shape[1]])

        if ~np.max(np.isnan(p_c)):
            p_c      = np.array(p_c).reshape(2, 1)
            d = dist(p_r, p_c).reshape(-1)
            p_r = p_r[:, d.argsort()[0:D.shape[1]]]
            val_r = val_r[:, d.argsort()[0:D.shape[1]]]
            # D = np.zeros([3, p_r.shape[1]])
            if ~np.max(np.isnan(ll)):
                D[2, ] = distline(p_r, ll) * w[2]
            else:
                D[2, ] = dist(p_c, p_r) * w[2]

        if ~np.max(np.isnan(p_old)):
            p_old    = np.array(p_old).reshape(2, 1)
            d = dist(p_r, p_old).reshape(-1)
            p_r = p_r[:, d.argsort()[0:D.shape[1]]]
            val_r = val_r[:, d.argsort()[0:D.shape[1]]]
            D[0, ] = dist(p_r, p_old) * w[0]


        if ~np.max(np.isnan(val_old)):
            val_old  = np.array(val_old).reshape(1, 1)
            D[1, ] = np.abs(val_r - val_old)
            D[1, ] = D[1, ]/np.max(D[1, ])*w[1]



        DD = np.sum(D**2, axis=0)
        p_r = p_r[:, DD.argsort()]
        val_r = val_r[:, DD.argsort()]
        DD = DD[DD.argsort()]


        if ~np.max(np.isnan(p_c)):
            X = p_c.reshape(2, 1)
            V = np.max(val_r).reshape(1, 1)
            P = 1

            for i in range(0, p_r.shape[1]):
                Y = p_r[:, i].reshape(2, 1)
                W = val_r[:, i]
                R = DD[i]
                K = P/(P+R)
                X += K*(Y - X)
                V += K*(W - V)
                P = (1-K)*P
        else:
            X = p_old.reshape(2, 1)
            V = val_old.reshape(1, 1)
            P = .1

            for i in range(p_r.shape[1]):
                Y = p_r[:, i].reshape(2, 1)
                W = val_r[:, i]
                R = DD[i]
                K = P / (P + R)
                X += K * (Y - X)
                V += K * (W - V)
                P = (1 - K) * P

        return X.reshape(2, 1), V
    else:
        p_r = np.array(p_r).reshape(2, -1)
        val_r = np.array(val_r).reshape(1, -1)
        D = np.zeros([3, p_r.shape[1]])
        if ~np.max(np.isnan(p_old)):
            p_old = np.array(p_old).reshape(2, 1)
            D[0,] = dist(p_r, p_old) * w[0]

        if ~np.max(np.isnan(val_old)):
            val_old = np.array(val_old).reshape(1, 1)
            D[1,] = np.abs(val_r - val_old)
            D[1,] = D[1,] / np.max(D[1,]) * w[1]

        if ~np.max(np.isnan(p_c)):
            p_c = np.array(p_c).reshape(2, 1)
            if ~np.max(np.isnan(ll)):
                D[2,] = distline(p_r, ll) * w[2]
            else:
                D[2,] = dist(p_c, p_r) * w[2]

        DD = np.sum(D, axis=0)
        return p_r[:, DD.argmin().reshape(-1)], val_r[:, DD.argmin().reshape(-1)]


def img_undo_filter(box, u_lim=[0, 1], v_lim=[0, 1]):
    box[:, [0, 2]] = (v_lim[1] - v_lim[0]) * box[:, [0, 2]] + v_lim[0]
    box[:, [1, 3]] = (u_lim[1] - u_lim[0]) * box[:, [1, 3]] + u_lim[0]
    return


def img_filter(img, u_lim=[0, 1], v_lim=[0, 1]):
    u = (np.array(u_lim)*img.shape[1]).astype(int)
    v = (np.array(v_lim)*img.shape[0]).astype(int)
    mask = np.zeros(img.shape[0:2], np.uint8)
    mask[v[0]:v[1], u[0]:u[1]] = 255
    res = cv2.bitwise_and(img, img, mask=mask)

    return res


def box_filter(box, u_lim=[0, 1], v_lim=[0, 1]):
    indicator = list(((box[:, 0] > v_lim[0]) &
                      (box[:, 1] > u_lim[0]) &
                      (box[:, 2] < v_lim[1]) &
                      (box[:, 3] < u_lim[1])
                      ).reshape(-1))
    return box[indicator, :]


def heat_map(tabold, xr, yr, zr, xlim, ylim, xc=np.nan, yc=np.nan, xbinnum=100, ybinnum=100):

    x_edges = np.linspace(xlim[0], xlim[1], xbinnum)
    y_edges = np.linspace(ylim[0], ylim[1], ybinnum)

    try:
        valid_list = np.logical_and(
            np.logical_and(xr >= xlim[0], xr <= xlim[1]),
            np.logical_and(yr >= ylim[0], yr <= ylim[1]))

        xr = xr[valid_list]
        yr = yr[valid_list]
        zr = zr[valid_list]
        # zr = np.log10(zr[valid_list]+1)
        # zr = np.ones(len(xr))

        zr = zr/np.max(zr)

        indx = np.digitize(xr, x_edges)
        indy = np.digitize(yr, y_edges)

        xr = x_edges[indx-1]
        yr = y_edges[indy-1]

        indx = np.digitize(xc, x_edges)
        indy = np.digitize(yc, y_edges)

        xc = x_edges[indx-1]
        yc = y_edges[indy-1]

        tab = np.zeros([xbinnum, ybinnum])

        for i in range(len(xr)):
            if yr[i]> 1:
                tab[np.where(x_edges == xr[i]), np.where(y_edges == yr[i])] =+ zr[i]

        try:
            for i in range(len(xc)):
                tab[np.where(x_edges == xc[i]), np.where(y_edges == yc[i])] =+ 1
        except:
            pass


        tabold = np.append(tab.reshape(xbinnum, ybinnum, 1), tabold, axis=-1)
        tabold = np.delete(tabold, -1, axis=-1)

        return x_edges, y_edges, tabold
    except:
        pass


def heat_map_polar(tabold, xr, yr, zr, rlim, philim, xc=np.nan, yc=np.nan, rbinnum=100, phibinnum=100):

    r_edges = np.linspace(rlim[0], rlim[1], rbinnum)
    phi_edges = np.linspace(philim[0], philim[1], phibinnum)

    rr = np.sqrt(xr**2+yr**2)
    phir = np.arctan(xr/yr)
    try:
        valid_list = np.logical_and(
            np.logical_and(rr >= rlim[0], rr <= rlim[1]),
            np.logical_and(phir >= philim[0], phir <= philim[1]))

        rr = rr[valid_list]
        phir = phir[valid_list]
        # zr = zr[valid_list]
        zr = np.log10(zr[valid_list]+1)
        # zr = np.ones(len(xr))

        # zr = zr/np.max(zr)

        indx = np.digitize(rr, r_edges)
        indy = np.digitize(phir, phi_edges)

        rr = r_edges[indx-1]
        phir = phi_edges[indy-1]

        indx = np.digitize(xc, r_edges)
        indy = np.digitize(yc, phi_edges)


        tab = np.zeros([rbinnum, phibinnum])

        for i in range(len(rr)):
            if rr[i]> 1.5:
                tab[np.where(r_edges == rr[i]), np.where(phi_edges == phir[i])] =+ zr[i]


        tabold = np.append(tab.reshape(rbinnum, phibinnum, 1), tabold, axis=-1)
        tabold = np.delete(tabold, -1, axis=-1)

        return r_edges, phi_edges, tabold
    except:
        pass


def res_var(d_old, m_new, n=10):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    x = np.linspace(1, n, n).reshape(1, -1)

    try:
        d_old = np.append([m_new], d_old, axis=-1)
        d_old = np.delete(d_old, -1, axis=-1)
        # regression_model = LinearRegression()
        # regression_model = regression_model.fit(x, d_old.reshape(1, -1))
        # y_predicted = regression_model.predict(x)
        # rmse = mean_squared_error(d_old.reshape(1, -1), y_predicted)
        rmse = np.var(d_old)
    except:
        rmse = 10

    return rmse, d_old


def update_measure(R, l_corner, w_trailer, var, var_t, missed_flag, missed_marker_light):#, origin):

    if var > var_t:
        l_corner_new = np.nanmean(np.sqrt(np.nansum(R ** 2, axis=0)))
        w_trailer_new = np.sqrt((R[0, 0] - R[0, 1]) ** 2 + (R[1, 0] - R[1, 1]) ** 2)

        if not np.isnan(l_corner) and not np.isnan(w_trailer):
            if not np.isnan(l_corner_new) and not np.isnan(w_trailer_new):
                l_corner  = (var * l_corner_new + l_corner) / (var + 1)
                w_trailer = (var * w_trailer_new + w_trailer) / (var + 1)


        else:
            l_corner = l_corner_new
            w_trailer = w_trailer_new


    l_trailer = np.sqrt(l_corner ** 2 - (w_trailer / 2) ** 2)
    omega = np.arcsin(w_trailer / l_corner / 2)*2


    if missed_flag:
        if missed_marker_light == 'p':
            phi = np.degrees(np.arctan(R[0, 1] / R[1, 1]) - omega / 2)
        elif missed_marker_light == 'd':
            phi = np.degrees(np.arctan(R[0, 0] / R[1, 0]) + omega / 2)
    else:

        c, s = np.cos(omega), np.sin(omega)

        p = R[:, 0]
        d = R[:, 1]


        TF = np.array([[c, s], [-s, c]]).reshape(2, 2)
        p_rotated = TF.dot(p)
        m = np.mean([p_rotated, d], axis=0)

        phi = np.degrees(np.arctan(m[0] / m[1]) - omega / 2)

    # if var > var_t and ~np.isnan(R).any():
    #     mid_point = np.mean([p, d], axis=0)
    #     alpha = np.arctan((p[1]-d[1])/(p[0]-d[0]))
    #     origin += np.array([[mid_point[0] - l_trailer * np.sin(np.radians(phi))],
    #                          [mid_point[1] - l_trailer * np.cos(np.radians(phi))]]).reshape(2, 1)

    return phi, l_corner, w_trailer#, origin


def tfcv_convertor(boxes, sensor_size, source='tf'):
    box = np.asarray(boxes)
    if source == 'tf':
        boxes_tf = de_normalize(boxes, sensor_size).copy()
        boxes = []
        for row in range(boxes_tf.shape[0]):
            box[row, 0] = boxes_tf[row, 1]
            box[row, 1] = boxes_tf[row, 0]
            box[row, 2] = boxes_tf[row, 3] - boxes_tf[row, 1]
            box[row, 3] = boxes_tf[row, 2] - boxes_tf[row, 0]
            bbox = tuple(box[row, :])
            boxes.append(bbox)

    elif source == 'cv':
        boxes_cv = np.asarray(boxes).copy()
        for row in range(boxes_cv.shape[0]):
            box[row, 0] = boxes_cv[row, 1]
            box[row, 1] = boxes_cv[row, 0]
            box[row, 2] = boxes_cv[row, 3] + boxes_cv[row, 1]
            box[row, 3] = boxes_cv[row, 2] + boxes_cv[row, 0]
        boxes = box.reshape(-1, 4)
        boxes = normalize(boxes, sensor_size)

    return boxes

