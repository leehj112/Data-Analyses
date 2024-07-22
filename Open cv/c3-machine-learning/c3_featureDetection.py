# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:28:01 2024

@author: leehj
"""

# 특징 검지 예제 (c3_featureDetection.py)

# 관련 라이브러리 선언
import numpy as np
import cv2
from imgRead import imgRead
from createFolder import createFolder

# 영상 읽기
img1 = imgRead("../images/img7.jpg", cv2.IMREAD_UNCHANGED, 320, 240)

# 특징 검지 및 결과 영상 저장
save_dir = '../code_res_imgs/c3_featureDetection'
createFolder(save_dir)
img1_gray= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

keyPoint = cv2.goodFeaturesToTrack(img1_gray, 25, 0.01, 10)
keyPoint = np.int0(keyPoint)
img2 = cv2.cvtColor(img1_gray, cv2.COLOR_GRAY2BGR)
for i in keyPoint:
    x,y = i.ravel()
    cv2.circle(img2, (x,y), 5, (0,0,255))
    cv2.imshow("goodToTrack", img2)
    cv2.imwrite(save_dir + "/goodToTrack.jpg", img2)

sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
fast = cv2.FastFeatureDetector_create()
orb = cv2.ORB_create()

methods = [(sift, 'sift'),
           (surf, 'surf'),
           (fast, 'fast'),
           (orb, 'orb')]
for (method, name) in methods:
    print(name)
    keyPoint = method.detect(img1_gray, None)
    res = cv2.drawKeypoints(img1_gray, keyPoint, img1)
    cv2.imshow(name, res)
    cv2.imwrite(save_dir + "/" + name + ".jpg", res)

# 키보드 입력을 기다린 후 모든 영상창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()