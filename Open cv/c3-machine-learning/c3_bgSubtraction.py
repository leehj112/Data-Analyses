# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:27:16 2024

@author: leehj
"""

# 전경-배경 분할 예제 (c3_bgSubtraction.py)

# 관련 라이브러리 선언
import numpy as np
import cv2
from imgRead import imgRead
from createFolder import createFolder
from matplotlib import pyplot as plt

save_dir = '../code_res_imgs/c3_bgSubtraction'
createFolder(save_dir)

# 동영상 불러오기
cap = cv2.VideoCapture('../images/video2.mp4')
if cap.isOpened() == False:
    print('Can not open the...')
    exit()
    
# 전경-배경 분할 수행자 선언
bgMethod1 = cv2.createBackgroundSubtractorMOG2()
bgMethod2 = cv2.createBackgroundSubtractorKNN()
bgMethod1_blur = cv2.createBackgroundSubtractorMOG2()
bgMethod2_blur = cv2.createBackgroundSubtractorKNN()
bgMethod3 = cv2.bgsegm.createBackgroundSubtractorCNT()
bgMethod4 = cv2.bgsegm.createBackgroundSubtractorGMG()
bgMethod5 = cv2.bgsegm.createBackgroundSubtractorMOG()

imgIndex = 1
while(cap.isOpened()):
    # 영상이 읽어오고, 존재하지 않으면 while 종료
    ret, frame = cap.read()
    if ret != True or frame is None:
        print("cap.read() break")
        break

    # 전경 배경 분리 수행
    frame = cv2.resize(frame, (320, 240))
    if imgIndex != 1:
        bgMOG = bgMethod1.getBackgroundImage()
        bgKNN = bgMethod2.getBackgroundImage()
        bgMOG_blur = bgMethod1_blur.getBackgroundImage()
        bgKNN_blur = bgMethod2_blur.getBackgroundImage()
    fgMOG = bgMethod1.apply(frame, learningRate = -1)
    fgKNN = bgMethod2.apply(frame)
    fgMOG_blur = bgMethod1_blur.apply(cv2.blur(frame, (5,5)), learningRate=-1)
    fgKNN_blur = bgMethod2_blur.apply(cv2.blur(frame, (5,5)))

    cv2.imshow('input frame', frame)
    cv2.imshow('fg using mog2', fgMOG)
    cv2.imshow('fg using knn', fgKNN)
    cv2.imshow('fg using mog2(blur)', fgMOG_blur)
    cv2.imshow('fg using knn(blur)', fgKNN_blur)
    if imgIndex != 1:
        cv2.imshow('bg knn', bgKNN)
        cv2.imshow('bg mog2', bgMOG)
        cv2.imshow('bg knn_blur', bgKNN_blur)
        cv2.imshow('bg mog2_blur', bgMOG_blur)
        
    key = cv2.waitKey(1)
    if key == 27:
        print('ESC break')
        break

    """
    # 영상 저장
    cv2.imwrite(save_dir + "/" + str(imgIndex) + "_in.jpg", frame)
    cv2.imwrite(save_dir + "/" + str(imgIndex) + "_mog2.jpg", fgMOG)
    cv2.imwrite(save_dir + "/" + str(imgIndex) + "_knn.jpg", fgKNN)
    cv2.imwrite(save_dir + "/" + str(imgIndex) + "_mog2_blur.jpg", fgMOG_blur)
    cv2.imwrite(save_dir + "/" + str(imgIndex) + "_knn_blur.jpg", fgKNN_blur)
    if imgIndex != 1:
        cv2.imwrite(save_dir + "/" + str(imgIndex) + "_bgMog2.jpg", bgMOG)
        cv2.imwrite(save_dir + "/" + str(imgIndex) + "_bgKnn.jpg", bgKNN)
        cv2.imwrite(save_dir + "/" + str(imgIndex) + "_bgMog2_blur.jpg", bgMOG_blur)
        cv2.imwrite(save_dir + "/" + str(imgIndex) + "_bgKnn_blur.jpg", bgKNN_blur)
    """
        
    imgIndex += 1


        
cap.release() # 카메라 해제

cv2.destroyAllWindows()