# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 13:20:51 2018

@author: Ravibalg
"""

import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np

extracted_text = []
# Load the classifier
clf = joblib.load("all_cls.pkl")

# Read the input image 
#im= cv2.imread("C:\\Users\\Ravibalg\\Desktop\\Somethingyoudontwannaknow\\python!\\Imageprc\\English\\Hnd\\Img\\Sample025\\img025-055.png")
im = cv2.imread("paint4.png")
# rotate the image by 180 degrees
# (h, w) = image.shape[:2]
# center = (w / 2, h / 2)
# M = cv2.getRotationMatrix2D(center, 90, 1.0)
# im = cv2.warpAffine(image, M, (w, h))

#resize dimemsions
r = 1500.0 / im.shape[1]
dim = (1500, int(im.shape[0] * r))

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
_,ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
for rect in rects:
    # Draw the rectangles
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
    # Make the rectangular region around the digit
    
    roi = im_th[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]
    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    # Calculate the HOG features
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
    #extracted_text.append(str((nbr[0])))
    cv2.putText(im, str((nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
 
# perform the actual resizing of the image and show it
#resized = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("resized", im)
cv2.waitKey(0)
#cv2.imshow("Resulting Image with Rectangular ROIs", im)
#cv2.waitKey()

