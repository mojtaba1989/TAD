'''
In this example we are saving the data to a csv file to be opened with Excel.

'''

from gdx import gdx 
gdx = gdx.gdx()

import csv
import time
import numpy as np

myFile = open('csvexample.csv', 'w', newline='') 
writer = csv.writer(myFile)

gdx.open_usb()
gdx.select_sensors([6])
gdx.start(period = 20) 
column_headers = gdx.enabled_sensor_info()
writer.writerow(['time',column_headers])

for i in range(0,20):
    measurements = gdx.read() 
    if measurements == None: 
        break
    writer.writerow([time.time(), np.degrees(measurements)])
    print(measurements)

gdx.stop()
gdx.close()