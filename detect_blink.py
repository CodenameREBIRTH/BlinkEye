import dlib
import imutils
from imutils.video import VideoStream
from imutils import face_utils
from scipy.spatial import distance as dist
import numpy as np
import os
import cv2 as cv
import time
import pyttsx3

# declare eye aspect ratio threshold and consecutive frames threshold
thresholdRatio = 0.3
thresholdFrame = 3

# intialize blinks counter and frame counter
blinkcount =0
framecount = 0

# calculate eye aspect ratio (ear)
def eye_aspect_ratio(eye):
    # compute horizontal distance
    h = 2.0 * dist.euclidean(eye[0], eye[3])
    # compute vertical distance
    v = dist.euclidean(eye[1], eye[5]) + dist.euclidean(eye[2], eye[4])
    # compute ratio
    ratio = v / h
    return ratio


# initialise dlib face detector (HOG - based) model
print("loading facial landmark detector model")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.sep.join(["BlinkDetector", "shape_predictor_68_face_landmarks.dat"]))

# get indexes for left and right eye facial landmark
(leftstart, leftend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightstart, rightend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# initialise video stream
print("starting camera")
cap = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = cap.read()
    frame = imutils.resize(frame, width=450)
    grayframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # detect faces in grayscale
    facedetected = detector(grayframe, 0)

    for face in facedetected:
        shape = predictor(grayframe, face)
        shape = face_utils.shape_to_np(shape)
        # get left and right eye coordinates
        lefteye = shape[leftstart:leftend]
        righteye = shape[rightstart:rightend]
        leftratio = eye_aspect_ratio(lefteye)
        rightratio  =eye_aspect_ratio(righteye)

        ratio = (leftratio + rightratio) / 2.0

        # compute convexhull for each eye aspect ratio
        leftconvexhull = cv.convexHull(lefteye)
        rightconvexhull = cv.convexHull(righteye)
        cv.drawContours(frame, [leftconvexhull], -1, (0, 255, 0), 1)
        cv.drawContours(frame, [rightconvexhull], -1, (0, 255, 0), 1)

        if ratio < thresholdRatio:
            framecount += 1
        else:
            if framecount >= thresholdFrame:
                blinkcount += 1
            framecount = 0

        cv.putText(frame, "Blinks Count: {}".format(blinkcount), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv.putText(frame, "Ratio: {}".format(ratio), (300, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # if curr != blinkcount:
        #     speech.say('Blink Count' + str(blinkcount))
        #     curr = blinkcount
    # show output and break loop when 'q' is pressed
    cv.imshow("Blink Detector", frame)
    if cv.waitKey(10) == ord('q'):
        break
# destroy all windows
cv.destroyAllWindows()
cap.stop()
speech = pyttsx3.init()
speech.say('Total blinks are' + str(blinkcount))
speech.runAndWait()
speech.stop()