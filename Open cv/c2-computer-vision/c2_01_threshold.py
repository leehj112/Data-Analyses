# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:58:32 2024

@author: leehj
"""

# 영상 이진화 예제 (c2_threshold.py)

# 검정과 흰색의 2가지 레벨로 표현

# 관련 라이브러리 선언
import numpy as np
import cv2
from imgRead import imgRead
from createFolder import createFolder

# 영상 읽기
img1 = imgRead("../images/img14.jpg", cv2.IMREAD_GRAYSCALE, 320, 240)

# 영상 이진화 수행 : 이진화방법
methods = [cv2.THRESH_BINARY,
           cv2.THRESH_BINARY_INV,
           cv2.THRESH_TRUNC,
           cv2.THRESH_TOZERO,
           cv2.THRESH_TOZERO_INV,
           cv2.THRESH_OTSU,
           cv2.THRESH_TRIANGLE,
           cv2.ADAPTIVE_THRESH_MEAN_C,      # 7
           cv2.ADAPTIVE_THRESH_GAUSSIAN_C]  # 8

thres = 70    # 임계치
maxVal = 255  # 특정방법
ress = []
for i in range(0, 7):
    ret, res = cv2.threshold(img1, thres, maxVal, methods[i])
    ress.append(res)
    print("ret:", ret)

# 입력 영상을 지정된 임계치와 방법을 이용하여 이진화    
ress.append(cv2.adaptiveThreshold(img1, maxVal, methods[7], methods[0], 61, 0)) # 산출평균
ress.append(cv2.adaptiveThreshold(img1, maxVal, methods[8], methods[0], 61, 0)) # 가우시안 분포

# 결과 영상 출력
displays = [("input1", img1),
            ("res1", ress[0]),
            ("res2", ress[1]),
            ("res3", ress[2]),
            ("res4", ress[3]),
            ("res5", ress[4]),
            ("res6", ress[5]),
            ("res7", ress[6]),
            ("res8", ress[7]),
            ("res9", ress[8])]
for (name, out) in displays:
    cv2.imshow(name, out)

# 키보드 입력을 기다린 후 모든 영상창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()

# 영상 저장
save_dir = '../code_res_imgs/c2_threshold'
createFolder(save_dir)
for (name, out) in displays:
    cv2.imwrite(save_dir+"/"+name+".jpg", out)

