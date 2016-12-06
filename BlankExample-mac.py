import cv2
import json
import time
import datetime

# load configuration file
config = json.load(open("config.json"))

cv2.namedWindow("Output")

# camera
capture = cv2.VideoCapture(0)
capture.set(3, 320)
capture.set(4, 240)
if capture.isOpened():
    rval, frame = capture.read()
else:
    rval = False
time.sleep(config["camera_warmup"])

# main loop
while rval:
    # get frame
    rval, frame = capture.read()
    cv2.imshow("Output", frame)

    # escape key
    key = cv2.waitKey(10)
    if key == 27:
        break

# cleanup
cv2.destroyWindow("Output")
