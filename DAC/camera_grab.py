from pypylon import pylon
import time

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

numberOfImagesToGrab = 100
camera.StartGrabbingMax(numberOfImagesToGrab)
img = pylon.PylonImage()
i = 0
while camera.IsGrabbing() and i < 10:
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if grabResult.GrabSucceeded():
        ipo = pylon.ImagePersistenceOptions()
        ipo.SetQuality(quality=100)

        filename = "saved_pypylon_img_%d.jpeg" % i
        img.Save(pylon.ImageFileFormat_Jpeg, filename, ipo)
        time.sleep(2)

    grabResult.Release()