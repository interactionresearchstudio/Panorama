import cv2
import json
import time
import datetime
import os

# configuration file
config = json.load(open("config.json"))
# end of configuration file

# window
cv2.namedWindow("Output")
# end of window

# camera
capture = cv2.VideoCapture(0)
capture.set(3, 320)
capture.set(4, 240)
if capture.isOpened():
    rval, frame = capture.read()
else:
    rval = False
time.sleep(config["camera_warmup"])
# end of camera

# main loop
while rval:
    # new frame
    rval, image = capture.read()
    # end of new frame

    cv2.imshow("Output", image)

    # wait for keys
    key = cv2.waitKey(10)
    if key == 27:
        break
    # end of loop

# cleanup
cv2.destroyWindow("Output")
