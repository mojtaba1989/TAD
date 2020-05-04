import os
import sys
import csv
import python_data_fusion as pdf
import numpy as np

if __name__ == '__main__':

    is_error = False
    for idx, arg in enumerate(sys.argv):
        if arg in ['--file', '-f']:
            FILE_NAME = str(sys.argv[idx+1])
            del sys.argv[idx]
            del sys.argv[idx]
        else:
            is_error = True

    CWD_PATH = os.getcwd()
    PATH_TO_CSV = os.path.join(CWD_PATH, FILE_NAME, 'TP-lined' + FILE_NAME + '.csv')

    for idx, arg in enumerate(sys.argv):
        if arg in ['--index', '-id']:
            INDEX = int(sys.argv[idx + 1])
            del sys.argv[idx]
            del sys.argv[idx]
        else:
            is_error = True

    if len(sys.argv) != 1:
        is_error = True
    else:
        for arg in sys.argv:
            if arg.startswith('-'):
                is_error = True

    x_error = .91651997
    y_error = -.71220556
    T_est_old = np.nan

    if 'INDEX' in locals():
        vernier, x_p, y_p, x_d, y_d, x_c, y_c, p_p, p_d, p_c = pdf.readTPCSV(PATH_TO_CSV, INDEX)

        print('vernier', vernier)
        print('cam', pdf.cam_ang_det_L(p_c))
    else:
        INDEX = 1
        myFile = open(os.path.join(CWD_PATH, FILE_NAME, 'TPDA-lined-cleaned-' + FILE_NAME + '.csv'), 'w', newline='')
        writer = csv.writer(myFile)
        writer.writerow(['Index', 'Vernier',
                         'X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 'X_8', 'X_9', 'X_10',
                         'X_11', 'X_12', 'X_13', 'X_14', 'X_15', 'X_16', 'X_17', 'X_18', 'X_19', 'X_20',
                         'Y_1', 'Y_2', 'Y_3', 'Y_4', 'Y_5', 'Y_6', 'Y_7', 'Y_8', 'Y_9', 'Y_10',
                         'Y_11', 'Y_12', 'Y_13', 'Y_14', 'Y_15', 'Y_16', 'Y_17', 'Y_18', 'Y_19', 'Y_20',
                         'T_1', 'T_2', 'T_3', 'T_4', 'T_5', 'T_6', 'T_7', 'T_8', 'T_9', 'T_10',
                         'T_11', 'T_12', 'T_13', 'T_14', 'T_15', 'T_16', 'T_17', 'T_18', 'T_19', 'T_20', 'T_est',
                         "d_T_est", "T_est_inv", "New_Ver"])
        while True:
            idx, vernier, x_p, y_p, x_d, y_d, x_c, y_c, p_p, p_d, p_c = pdf.readTPCSV(PATH_TO_CSV, INDEX)
            x_c = np.array(x_c)
            y_c = np.array(y_c)

            x_c = np.where(x_c == x_error, np.nan, x_c)
            y_c = np.where(y_c == y_error, np.nan, y_c)


            y_c = y_c[x_c.argsort()]
            x_c.sort()

            T = np.degrees(np.arctan(x_c/(y_c-.29)))
            T_est = np.mean(T[~np.isnan(T)])
            d_T_est = T_est - T_est_old
            T_est_old = T_est
            T_est_inv = 1/abs(T_est/40)
            New_Ver = np.array(vernier) - 5

            try:
                writer.writerow([idx, np.array(vernier).item(0),
                                 np.array(x_c).item(0), np.array(x_c).item(1),
                                 np.array(x_c).item(2), np.array(x_c).item(3),
                                 np.array(x_c).item(4), np.array(x_c).item(5),
                                 np.array(x_c).item(6), np.array(x_c).item(7),
                                 np.array(x_c).item(8), np.array(x_c).item(9),
                                 np.array(x_c).item(10), np.array(x_c).item(11),
                                 np.array(x_c).item(12), np.array(x_c).item(13),
                                 np.array(x_c).item(14), np.array(x_c).item(15),
                                 np.array(x_c).item(16), np.array(x_c).item(17),
                                 np.array(x_c).item(18), np.array(x_c).item(19),
                                 np.array(y_c).item(0), np.array(y_c).item(1),
                                 np.array(y_c).item(2), np.array(y_c).item(3),
                                 np.array(y_c).item(4), np.array(y_c).item(5),
                                 np.array(y_c).item(6), np.array(y_c).item(7),
                                 np.array(y_c).item(8), np.array(y_c).item(9),
                                 np.array(y_c).item(10), np.array(y_c).item(11),
                                 np.array(y_c).item(12), np.array(y_c).item(13),
                                 np.array(y_c).item(14), np.array(y_c).item(15),
                                 np.array(y_c).item(16), np.array(y_c).item(17),
                                 np.array(y_c).item(18), np.array(y_c).item(19),
                                 T.item(0), T.item(1),
                                 T.item(2), T.item(3),
                                 T.item(4), T.item(5),
                                 T.item(6), T.item(7),
                                 T.item(8), T.item(9),
                                 T.item(10), T.item(11),
                                 T.item(12), T.item(13),
                                 T.item(14), T.item(15),
                                 T.item(16), T.item(17),
                                 T.item(18), T.item(19),
                                 T_est.item(0), d_T_est.item(0),
                                 T_est_inv.item(0), New_Ver.item(0)])
            except:
                pass

            INDEX += 1
