# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:11:37 2024

@author: leehj
"""

# 옵티클 플로우 예제 (c3_opticalFlow.py)

# 관련 라이브러리 선언
import cv2
import numpy as np
from createFolder import createFolder

save_dir = "./code_res_imgs/c3_opticalFlow"
createFolder(save_dir)

# feature point 기반 optical flow 방법
# 비디오 읽기
cap = cv2.VideoCapture("./images/video7.mp4")

# ShiTomasi 코너 검출기 변수 설정
numMaxCorners = 20
feature_params = dict( maxCorners = numMaxCorners,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# 루카스-카나데 기반 옵티클 플로우 변수 설정
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 초기 프레임에서 특징점 검출
ret, prevFrame = cap.read()
prevFrame = cv2.resize(prevFrame, (320, 240))
prevGray = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(prevGray, mask = None, **feature_params)

# 옵티클 플로우를 표현하기 위한 색 및 이미지 생성
flowColor = np.random.randint(0,255,(numMaxCorners,3))
mask = np.zeros_like(prevFrame)

indexImg = 1
while(1):
    ret, curFrame = cap.read()
    if ret != True:
        break

    curFrame = cv2.resize(curFrame, (320, 240))
    curGgray = cv2.cvtColor(curFrame, cv2.COLOR_BGR2GRAY)

    # 옵티클 플로우 계산
    p1, st, err = cv2.calcOpticalFlowPyrLK(prevGray, curGgray, p0, None, **lk_params)

    # good points 선택
    good_new = p1[st==1]
    good_old = p0[st==1]

    # 특징점 및 기준 프레임 업데이트
    prevGray = curGgray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # 특징점 추적 결과 표시
    for i, (new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), flowColor[i].tolist(), 2)
        curFrame = cv2.circle(curFrame,(a,b),5,flowColor[i].tolist(),-1)
    res = cv2.add(curFrame,mask)
    cv2.imshow('frame', res)
    cv2.imwrite(save_dir + "/res_lk_" + str(indexImg) + ".jpg", res)
    indexImg += 1

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# dense optical flow 방법
cap = cv2.VideoCapture("./images/video7.mp4")
ret, prevFrame = cap.read()
prevFrame = cv2.resize(prevFrame, (320, 240))
prvs = cv2.cvtColor(prevFrame,cv2.COLOR_BGR2GRAY)

# 옵티클 플로우를 표현하기 위한 이미지 생성
maskHSV = np.zeros_like(prevFrame)
maskHSV[...,1] = 255

# Farneback 기반 옵티클 플로우 변수 설정
fb_params = dict( pyr_scale  = 0.5,
                  levels = 3,
                  winsize = 15,
                  iterations = 3,
                  poly_n = 5,
                  poly_sigma = 1.2,
                  flags = 0)
indexImg = 1
while(1):
    ret, curFrame = cap.read()
    if ret != True:
        break

    curFrame = cv2.resize(curFrame, (320, 240))
    next = cv2.cvtColor(curFrame,cv2.COLOR_BGR2GRAY)

    # 옵티클 플로우 계산
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, **fb_params)

    # 기준 프레임 업데이트
    prvs = next

    # 옵티컬 플로우 결과 표시
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    maskHSV[...,0] = ang*180/np.pi/2
    maskHSV[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    res = cv2.cvtColor(maskHSV,cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2',res)
    cv2.imwrite(save_dir + "/res_dense_" + str(indexImg) + ".jpg", res)
    indexImg += 1

    # ESC 버튼 입력시 종료
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()