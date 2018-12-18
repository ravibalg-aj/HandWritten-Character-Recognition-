# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 14:14:40 2018

@author: RaviBalg
"""

from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
import cv2

trial_labels = []
for i in range(0,10):
    trial_labels.append(str(i))


for i in range(0,26):
    trial_labels.append(chr(65+i))


for i in range(0,26):
    trial_labels.append(chr(97+i))

labels = []
hog_features = []
for i in range(1,63):
    a = "{0:0=2d}".format(i)
    for j in range(1,56):
        b = "{0:0=2d}".format(j)
        
        #read_image
        im = cv2.imread("C:\\Users\\Ravibalg\\Desktop\\Somethingyoudontwannaknow\\python!\\Imageprc\\English\\Hnd\\Img\\Sample0"+str(a)+"\\img0"+str(a)+"-0"+str(b)+".png")
        
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
            #leng = int(rect[3] * 1.6)
            #pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            #pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
            roi = im_th[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]
            # Resize the image
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))
            #cv2.imshow("Roi",roi)
            features,hog_image = hog(roi,orientations=9,pixels_per_cell=(14,14),cells_per_block=(1,1),visualise=True)
        labels.append(trial_labels[(i-1)])
        hog_features.append(features)
       
from sklearn.ensemble import RandomForestClassifier       
clf=RandomForestClassifier(n_estimators=100,verbose=1)

from sklearn.model_selection import GridSearchCV
param={'n_estimators':[10,100,500]}
param1={'C':[0.1,0.001,0.5,1,2],'multi_class':['ovr','crammer_singer']}
gc=GridSearchCV(estimator=clf,param_grid=param,cv=10)
gc.fit(hog_features,labels)
gc.best_estimator_
gc.best_score_

joblib.dump(gc, "all_cls.pkl", compress=3)