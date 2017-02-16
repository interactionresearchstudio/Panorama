import cv2
import json
import time
import datetime
import os
import numpy as np
import imutils

class Stitcher:
    def __init__(self):
        self.isv3 = imutils.is_cv3()

    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        #unpack images, detect keypoints and extract local invariant descriptors
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # match features
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        if M is None:
            # not enough matched keypoints
            return None

        # otherwise, apply perspective warp
        (matches, H, status) = M
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # visualisation option
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            return (result, vis)

        return result

    def detectAndDescribe(self, image):
        # convert to gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # detect and extract features
        descriptor = cv2.xfeatures2d.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image, None)
        
        # convert keypoints to numpy arrays
        kps = np.float32([kp.pt for kp in kps])

        # return keypoints and features
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # compute raw matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        for m in rawMatches:
            # ensure close distance
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        if len(matches) > 4:
            # construct sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            
            # compute homography
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            return (matches, H, status)
        
        # otherwise, uncomputed
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # visualisation image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                # draw match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0,255,0), 1)
        
        return vis

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

stitcher = Stitcher()
imageA = None
imageB = None
vis = None
result = None

# main loop
while rval:
    # new frame
    rval, image = capture.read()
    # end of new frame

    cv2.imshow("Output", image)
    
    if imageA is not None:
        cv2.imshow("image a", imageA)
    if imageB is not None:
        cv2.imshow("image b", imageB)
    if result is not None:
        cv2.imshow("result", result)
    if vis is not None:
        cv2.imshow("matches", vis)
    

    # wait for keys
    key = cv2.waitKey(10)
    if key == ord('a'):
        imageA = image
    if key == ord('b'):
        imageB = image
    if key == ord('r'):
        (result, vis) = stitcher.stitch([imageB, imageA], showMatches=True)
    if key == 27:
        break
    # end of loop

# cleanup
cv2.destroyWindow("Output")
