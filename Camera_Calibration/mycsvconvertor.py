import os
import pandas as pd
import glob

file_list  =  glob.glob('*.csv')
CWD_PATH = os.getcwd()

print(file_list)

for fname in file_list:
    if "DSRC" in fname:
        Data = pd.read_csv(fname, sep=',')
        Data = Data['Time', 'Received', "Distance", "Latency", "RxPowerA", "RxPowerB"]
        Data.to_csv(os.path.join(CWD_PATH, fname))

    if "CV2X" in fname:
        Data = pd.read_csv(fname, sep=',')
        Data = Data['Time', 'Received', "Distance", "Latency"]
        Data.to_csv(os.path.join(CWD_PATH, fname))
