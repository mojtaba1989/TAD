# Packages

## General
import time
from time import gmtime, strftime, localtime
import numpy as np
import csv
import os

## RADAR
import serial
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import matplotlib.pyplot as plt

## Vernier
from gdx import gdx
gdx = gdx.gdx()

## Camera
from pypylon import pylon
import platform


# Set Directory and Create CSV
os.chdir('/Users/moj/TAD/')
dirpath, dirname = createFolder()
myFile = open(dirname+'.csv', 'w', newline='')
writer = csv.writer(myFile)
writer.writerow(['Index', 'time', column_header])

# Sensor Setup

## RADAR
# Change the configuration file name
configFileName = '1642config.cfg'
CLIportP = {}
DataportP = {}
CLIportD = {}
DataportD = {}


CLIportD, DataportD, CLIportP, DataportP = serialConfig(configFileName)

# Get the configuration parameters from the configuration file
configParameters = parseConfigFile(configFileName)

# START QtAPPfor the plot
app = QtGui.QApplication([])

## Vernier
gdx.open_usb()
gdx.select_sensors([6])
gdx.start(period=20)
column_header = gdx.enabled_sensor_info()

## Camera
"""
Available image formats are     (depending on platform):
 - pylon.ImageFileFormat_Bmp    (Windows)
 - pylon.ImageFileFormat_Tiff   (Linux, Windows)
 - pylon.ImageFileFormat_Jpeg   (Windows)
 - pylon.ImageFileFormat_Png    (Linux, Windows)
 - pylon.ImageFileFormat_Raw    (Windows)
"""
img = pylon.PylonImage()
tlf = pylon.TlFactory.GetInstance()

cam = pylon.InstantCamera(tlf.CreateFirstDevice())
cam.Open()
cam.StartGrabbing()
ipo = pylon.ImagePersistenceOptions()
ipo.SetQuality(quality=100)

#___________________Main__________________________
detObjD = {}
detObjP = {}
frameDataD = {}
frameDataP = {}
currentIndex = 0
while True:
    try:
        # Update Radar data and check if the data is okay

        dataOkD, dataOkP = update()

        if dataOkD and dataOkP:
            # Store the current frame into frameData
            frameDataD[currentIndex] = detObjD
            frameDataP[currentIndex] = detObjP
        # Update Vernier data and check if the data is okay
        measurements = gdx.read()

        # time.sleep(0.033)  # Sampling frequency of 30 Hz

        # Update Camera data
        with cam.RetrieveResult(2000) as result:
            # Calling AttachGrabResultBuffer creates another reference to the
            # grab result buffer. This prevents the buffer's reuse for grabbing.
            img.AttachGrabResultBuffer(result)
            filename = "img_%d.jpeg" % currentIndex
            img.Save(pylon.ImageFileFormat_Jpeg, filename, ipo)
            # In order to make it possible to reuse the grab result for grabbing
            # again, we have to release the image (effectively emptying the
            # image object).
            img.Release()

        currentIndex += 1

    # Stop the program and close everything if Ctrl + c is pressed
    except KeyboardInterrupt:
        CLIportD.write(('sensorStop\n').encode())
        CLIportD.close()
        DataportD.close()
        CLIportP.write(('sensorStop\n').encode())
        CLIportP.close()
        DataportP.close()
        win.close()
        break



# Closing
## RADAR

CLIportD.write(('sensorStop\n').encode())
CLIportD.close()
DataportD.close()
CLIportP.write(('sensorStop\n').encode())
CLIportP.close()
DataportP.close()
win.close()


## Vernier
gdx.stop()
gdx.close()

## Camera
cam.StopGrabbing()
cam.Close()


# Functions

def createFolder():
    try:
        dirpath = os.getcwd()
        dirname = strftime("%d-%b-%Y-%H-%M", localtime())
        os.makedirs(dirname)
        dirpath += '/'+dirname
        os.chdir(dirpath)
        os.makedirs('./Figures/')
    except OSError:
        print('Error in making directory')
    return dirpath, dirname


# Function to configure the serial ports and send the data from
# the configuration file to the radar

def serialConfig(configFileName):
    global CLIportP
    global DataportP
    global CLIportD
    global DataportD
    # Open the serial ports for the configuration and the data ports

    # Raspberry pi
    # CLIport = serial.Serial('/dev/tty.usbmodem', R00410274)
    # Dataport = serial.Serial('/dev/tty.usbmodem', R00410274)

    # Windows
    CLIportP = serial.Serial('COM3', 115200)
    DataportP = serial.Serial('COM4', 921600)
    CLIportD = serial.Serial('COM3', 115200)
    DataportD = serial.Serial('COM4', 921600)

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        CLIportP.write((i + '\n').encode())
        CLIportD.write((i + '\n').encode())
        print(i)
        time.sleep(0.01)

    return CLIportD, DataportD, CLIportP, DataportP

# Function to parse the data inside the configuration file
def parseConfigFile(configFileName):
    configParameters = {}  # Initialize an empty dictionary to store the configuration parameters

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:

        # Split the line
        splitWords = i.split(" ")

        # Hard code the number of antennas, change if other configuration is used
        numRxAnt = 6
        numTxAnt = 2

        # Get the information about the profile configuration
        if "profileCfg" in splitWords[0]:
            startFreq = int(splitWords[2])
            idleTime = int(splitWords[3])
            rampEndTime = int(splitWords[5]);
            freqSlopeConst = int(splitWords[8]);
            numAdcSamples = int(splitWords[10]);
            numAdcSamplesRoundTo2 = 1;

            while numAdcSamples > numAdcSamplesRoundTo2:
                numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2;

            digOutSampleRate = int(splitWords[11]);

        # Get the information about the frame configuration
        elif "frameCfg" in splitWords[0]:

            chirpStartIdx = int(splitWords[1]);
            chirpEndIdx = int(splitWords[2]);
            numLoops = int(splitWords[3]);
            numFrames = int(splitWords[4]);
            framePeriodicity = int(splitWords[5]);

    # Combine the read data to obtain the configuration parameters
    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * numAdcSamples)
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
    configParameters["dopplerResolutionMps"] = 3e8 / (2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate) / (2 * freqSlopeConst * 1e3)
    configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)

    return configParameters


# Funtion to read and parse the incoming data
def readAndParseData16xx(Dataport, configParameters):
    # global byteBuffer, byteBufferLength
    byteBuffer = np.zeros(2 ** 15, dtype='uint8')
    byteBufferLength = 0;

    # Constants
    OBJ_STRUCT_SIZE_BYTES = 12;
    BYTE_VEC_ACC_MAX_SIZE = 2 ** 15;
    MMWDEMO_UART_MSG_DETECTED_POINTS = 1;
    MMWDEMO_UART_MSG_RANGE_PROFILE = 2;
    maxBufferSize = 2 ** 15;
    magicWord = [2, 1, 4, 3, 6, 5, 8, 7]

    # Initialize variables
    magicOK = 0  # Checks if magic number has been read
    dataOK = 0  # Checks if the data has been read correctly
    frameNumber = 0
    detObj = {}

    readBuffer = Dataport.read(Dataport.in_waiting)
    byteVec = np.frombuffer(readBuffer, dtype='uint8')
    byteCount = len(byteVec)

    # Check that the buffer is not full, and then add the data to the buffer
    if (byteBufferLength + byteCount) < maxBufferSize:
        byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec[:byteCount]
        byteBufferLength = byteBufferLength + byteCount

    # Check that the buffer has some data
    if byteBufferLength > 16:

        # Check for all possible locations of the magic word
        possibleLocs = np.where(byteVec == magicWord[0])[0]

        # Confirm that is the beginning of the magic word and store the index in startIdx
        startIdx = []
        for loc in possibleLocs:
            check = byteVec[loc:loc + 8]
            if np.all(check == magicWord):
                startIdx.append(loc)

        # Check that startIdx is not empty
        if startIdx:

            # Remove the data before the first start index
            if startIdx[0] > 0:
                byteBuffer[:byteBufferLength - startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]
                byteBufferLength = byteBufferLength - startIdx[0]

            # Check that there have no errors with the byte buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0

            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

            # Read the total packet length
            totalPacketLen = np.matmul(byteBuffer[12:12 + 4], word)

            # Check that all the packet has been read
            if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                magicOK = 1

    # If magicOK is equal to 1 then process the message
    if magicOK:
        # word array to convert 4 bytes to a 32 bit number
        word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

        # Initialize the pointer index
        idX = 0

        # Read the header
        magicNumber = byteBuffer[idX:idX + 8]
        idX += 8
        version = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x')
        idX += 4
        totalPacketLen = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        platform = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x')
        idX += 4
        frameNumber = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        timeCpuCycles = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        numDetectedObj = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        numTLVs = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        subFrameNumber = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4

        # Read the TLV messages
        for tlvIdx in range(numTLVs):

            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

            # Check the header of the TLV message
            tlv_type = np.matmul(byteBuffer[idX:idX + 4], word)
            idX += 4
            tlv_length = np.matmul(byteBuffer[idX:idX + 4], word)
            idX += 4

            # Read the data depending on the TLV message
            if tlv_type == MMWDEMO_UART_MSG_DETECTED_POINTS:

                # word array to convert 4 bytes to a 16 bit number
                word = [1, 2 ** 8]
                tlv_numObj = np.matmul(byteBuffer[idX:idX + 2], word)
                idX += 2
                tlv_xyzQFormat = 2 ** np.matmul(byteBuffer[idX:idX + 2], word)
                idX += 2

                # Initialize the arrays
                rangeIdx = np.zeros(tlv_numObj, dtype='int16')
                dopplerIdx = np.zeros(tlv_numObj, dtype='int16')
                peakVal = np.zeros(tlv_numObj, dtype='int16')
                x = np.zeros(tlv_numObj, dtype='int16')
                y = np.zeros(tlv_numObj, dtype='int16')
                z = np.zeros(tlv_numObj, dtype='int16')

                for objectNum in range(tlv_numObj):
                    # Read the data for each object
                    rangeIdx[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                    idX += 2
                    dopplerIdx[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                    idX += 2
                    peakVal[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                    idX += 2
                    x[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                    idX += 2
                    y[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                    idX += 2
                    z[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                    idX += 2

                # Make the necessary corrections and calculate the rest of the data
                rangeVal = rangeIdx * configParameters["rangeIdxToMeters"]
                dopplerIdx[dopplerIdx > (configParameters["numDopplerBins"] / 2 - 1)] = dopplerIdx[dopplerIdx > (
                            configParameters["numDopplerBins"] / 2 - 1)] - 65535
                dopplerVal = dopplerIdx * configParameters["dopplerResolutionMps"]
                # x[x > 32767] = x[x > 32767] - 65536
                # y[y > 32767] = y[y > 32767] - 65536
                # z[z > 32767] = z[z > 32767] - 65536
                x = x / tlv_xyzQFormat
                y = y / tlv_xyzQFormat
                z = z / tlv_xyzQFormat

                # Store the data in the detObj dictionary
                detObj = {"numObj": tlv_numObj, "rangeIdx": rangeIdx, "range": rangeVal, "dopplerIdx": dopplerIdx, \
                          "doppler": dopplerVal, "peakVal": peakVal, "x": x, "y": y, "z": z}

                dataOK = 1

                # print(detObj['range'].mean())

            elif tlv_type == MMWDEMO_UART_MSG_RANGE_PROFILE:
                idX += tlv_length

        # Remove already processed data
        if idX > 0 and dataOK == 1:
            shiftSize = idX

            byteBuffer[:byteBufferLength - shiftSize] = byteBuffer[shiftSize:byteBufferLength]
            byteBufferLength = byteBufferLength - shiftSize

            # Check that there are no errors with the buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0

    return dataOK, frameNumber, detObj


# Funtion to update the data and display in the plot
def update():
    dataOkD = 0
    global detObjD
    xD = []
    yD = []
    dataOkP = 0
    global detObjP
    xP = []
    yP = []
    # Read and parse the received data
    dataOkD, frameNumberD, detObjD = readAndParseData16xx(DataportD, configParameters)
    dataOkP, frameNumberP, detObjP = readAndParseData16xx(DataportP, configParameters)
    if dataOkD:
        # print(detObj)
        xD = -detObjD["x"]
        yD = detObjD["y"]
    if dataOkP:
        xP = -detObjP["x"]
        yP = detObjP["y"]

    s.setData(xD, yD)
    s.setData(xP, yP)

    QtGui.QApplication.processEvents()

    return dataOkD, dataOkP