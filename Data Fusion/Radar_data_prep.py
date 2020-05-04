import os
import python_data_fusion as pdf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time




# FILE_NAME = '01-Oct-2019-13-56'
FILE_NAME = '20-Nov-2019-11-06'
# FILE_NAME = "04-Feb-2020-12-37"

CWD_PATH = os.getcwd()
PATH_TO_CSV = os.path.join(CWD_PATH, FILE_NAME, 'RADAR-lined-' + FILE_NAME + '.csv')

RESULT_DIR = os.path.join(CWD_PATH, FILE_NAME, "Result")
if not os.path.exists(RESULT_DIR):
    os.mkdir(RESULT_DIR)

x_error = .91651997
y_error = -.71220556
T_est_old = np.nan
p_r_old = np.nan
p_val_old = np.nan
d_r_old = np.nan
d_val_old = np.nan
RADAR_est_old = np.nan





center = np.array([[-.1], [.28]])

center_flag = True

pick_cam = np.zeros([2, 2])
pick_hist = np.zeros([2, 2])
pick_val = np.zeros([2, 2])

# W = [W_L, W_V, W_C]
w = [1, 1, 10]

Data = pd.read_csv(PATH_TO_CSV, sep=',')
PATH_TO_RESULTS = os.path.join(CWD_PATH, FILE_NAME, 'RADAR-cleaned-' + FILE_NAME + '.csv')



columns = ['Vernier', "Cam", "Radar", "L", "Lambda", "Hitch-Angle", "e(Cam)", "e(Radar)", "e(Hitch-Angle)"]



Results = pd.DataFrame(np.zeros([Data.__len__(), 9]), columns=columns)


plt.ion()

rdata = np.zeros([2])
cdata = np.zeros([2])

cam_l_corner = np.nan
cam_w_trailer = np.nan
cam_var = 1

radar_l_corner = np.nan
radar_w_trailer = np.nan
radar_var = 1




for INDEX in range(Data.__len__()):

    tstart = time.time()

    Results.loc[INDEX, "Vernier"] = np.array(pdf.csvCellReader(Data.loc[INDEX, "vernier"]))
    
    # CAMERA
    
    x_c = np.array(pdf.csvCellReader(Data.loc[INDEX, "Camera X"]))
    y_c = np.array(pdf.csvCellReader(Data.loc[INDEX, "Camera Y"]))

    x_c = np.where(x_c == x_error, np.nan, x_c)
    y_c = np.where(y_c == y_error, np.nan, y_c)


    y_c = y_c[x_c.argsort()]
    x_c.sort()
    center_flag = True

    y_c = y_c - center[1]
    x_c = x_c - center[0]



    try:
        p_c = np.array([[x_c[0]], [y_c[0]]])
        d_c = np.array([[x_c[19]], [y_c[19]]])
        if not np.isnan(x_c[19]) and not np.isnan(y_c[19]):
            R = np.concatenate([p_c, d_c], axis=1)

            cam_phi, cam_l_corner, cam_w_trailer, cam_disp = pdf.update_measure(
                R, cam_l_corner, cam_w_trailer, cam_var)

            cam_var, cdata = pdf.res_var(cdata, cam_phi, n=2)


            Results.loc[INDEX, "Cam"] = cam_phi
            Results.loc[INDEX, "e(Cam)"] = Results.loc[INDEX, "Vernier"] - cam_phi

        else:
            p_c = np.nan
            d_c = np.nan
            Results.loc[INDEX, "Cam"] = np.nan
            Results.loc[INDEX, "e(Cam)"] = np.nan
            cam_var = 10000
            cam_phi = 0
            cam_w_trailer = 0
            cam_l_corner = 0
    except:
        p_c = np.nan
        d_c = np.nan
        Results.loc[INDEX, "Cam"] = np.nan
        Results.loc[INDEX, "e(Cam)"] = np.nan
        cam_var = 10000
        cam_phi = 0
        cam_w_trailer = 0
        cam_l_corner = 0


    x_p = np.array(pdf.csvCellReader(Data.loc[INDEX, "RADAR X Passenger"])).reshape(1, -1) - center[0]
    y_p = np.array(pdf.csvCellReader(Data.loc[INDEX, "RADAR Y Passenger"])).reshape(1, -1) - center[1]
    p_p = np.concatenate([x_p, y_p], axis=0)
    x_d = np.array(pdf.csvCellReader(Data.loc[INDEX, "RADAR X Driver"])).reshape(1, -1) - center[0]
    y_d = np.array(pdf.csvCellReader(Data.loc[INDEX, "RADAR Y Driver"])).reshape(1, -1) - center[1]
    p_d = np.concatenate([x_d, y_d], axis=0)
    val_p = np.array(pdf.csvCellReader(Data.loc[INDEX, "RADAR Val Passenger"])).reshape(-1)
    val_d = np.array(pdf.csvCellReader(Data.loc[INDEX, "RADAR Val Driver"])).reshape(-1)


    p_r, p_val = pdf.Cam_Radar(
        p_p, val_p, p_r_old, p_val_old, p_c, w, method='kalman')
    d_r, d_val = pdf.Cam_Radar(
        p_d, val_d, d_r_old, d_val_old, d_c, w, method='kalman')

    R = np.concatenate([p_r, d_r], axis=1)

    radar_phi, radar_l_corner, radar_w_trailer, radar_disp = pdf.update_measure(
        R, radar_l_corner, radar_w_trailer, radar_var)

    radar_var, rdata = pdf.res_var(rdata, radar_phi, n=2)

    Results.loc[INDEX, "Radar"] = radar_phi
    Results.loc[INDEX, "e(Radar)"] = Results.loc[INDEX, "Vernier"] - radar_phi
    Results.loc[INDEX, "L"] = radar_l_corner
    Results.loc[INDEX, "Lambda"] = radar_w_trailer


    phi = (radar_phi * cam_var**2 + cam_phi * radar_var**2) / (cam_var**2 + radar_var**2)
    disp = (radar_disp * cam_var**2 + cam_disp * radar_var**2) / (cam_var**2 + radar_var**2)
    l_corner = (radar_l_corner * cam_var**2 + cam_l_corner * radar_var**2) / (cam_var**2 + radar_var**2)
    w_trailer = (radar_w_trailer * cam_var**2 + cam_w_trailer * radar_var**2) / (cam_var**2 + radar_var**2)
    l_trailer = np.sqrt(l_corner**2 - w_trailer**2/4)


    Results.loc[INDEX, "Hitch-Angle"] = phi
    Results.loc[INDEX, "e(Hitch-Angle)"] = Results.loc[INDEX, "Vernier"] - phi

    pose = np.array([[0, 0, w_trailer/2, -w_trailer/2], [0, l_trailer, l_trailer, l_trailer]])
    phi = np.radians(phi)
    c, s = np.cos(phi), np.sin(phi)
    TF = np.array(((c, s), (-s, c))).reshape(2, 2)
    pose = TF.dot(pose)


    #
    # if INDEX > 3:
    #     if ~np.all(np.isnan(disp)):
    #         center = center + cam_disp

    if np.mod(INDEX, 50) == 0:
        try:
            plt.scatter(p_p[0, ], p_p[1, ], label='PC-Passenger')
            plt.scatter(p_d[0, ], p_d[1, ], label='PC-Driver')
            plt.plot(x_c[[0, -1], ], y_c[[0, -1], ], label='CAM')
            plt.scatter(p_r[0, ], p_r[1, ], marker='o', edgecolors='b', facecolor='none', s=128, label='Picked PC-Passenger')
            plt.scatter(d_r[0, ], d_r[1, ], marker='o', edgecolors='r', facecolor='none', s=128, label='Picked PC-Driver')
            plt.plot(pose[0, 0:2], pose[1, 0:2])
            plt.plot(pose[0, 2:4], pose[1, 2:4], label="Trailer")
        except:
            pass


        plt.rcParams["figure.figsize"] = (5, 5)
        plt.legend(loc='lower center', ncol=3)
        plt.axis('equal')
        plt.ylim([0, 3])
        plt.xlim([-1.5, 1.5])
        plt.title("Epoch %d" %INDEX)
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")

        plt.show(block=False)
        plt.pause(.1)
        # plt.close()

        FIG_NAME = "epoch_%d.eps" %INDEX

        plt.savefig(os.path.join(RESULT_DIR, FIG_NAME), format='eps')

        plt.close()



    p_r_old = p_r
    p_val_old = p_val
    d_r_old = d_r
    d_val_old = d_val

    tend = time.time()
    Results.loc[INDEX, "Time"] = tend-tstart



Results = Results.dropna()
Results.to_csv(os.path.join(CWD_PATH, FILE_NAME, 'RADAR-cleaned-' + FILE_NAME + '.csv'), index=True)
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(Results["Vernier"], label='Vernier')
line2, = ax1.plot(Results["Cam"], label='Cam')
line3, = ax1.plot(Results["Hitch-Angle"], label='Hitch-Angle')
line7, = ax1.plot(Results["Radar"], label='RADAR')


ax1.legend()
plt.show()
plt.waitforbuttonpress()

print(center)
