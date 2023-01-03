import numpy as np
import cv2

def camera_correction(frame,a):
    width  = frame.shape[1]
    height = frame.shape[0]

    distCoeff = np.zeros((4,1),np.float64)

    # add your coefficients here!
    k1 = a; # negative to remove barrel distortion #-0.000012
    k2 = 0.0;
    p1 = 0.0;
    p2 = 0.0;

    distCoeff[0,0] = k1;
    distCoeff[1,0] = k2;
    distCoeff[2,0] = p1;
    distCoeff[3,0] = p2;

    # assume unit matrix for camera
    cam = np.eye(3,dtype=np.float32)

    cam[0,2] = width/2.0  # define center x
    cam[1,2] = height/2.0 # define center y
    cam[0,0] = 10.        # define focal length x
    cam[1,1] = 10.        # define focal length y

    # here the undistortion will be computed
    frame = cv2.undistort(frame,cam,distCoeff)
    # output = cv2.subtract(frame, dst) 
    # cv2.imshow('output',output)
    # cv2.imshow('dst',dst)
    # cv2.waitKey(0)
    return frame, cam, distCoeff
