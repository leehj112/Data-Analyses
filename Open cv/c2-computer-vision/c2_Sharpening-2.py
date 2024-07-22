# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:00:00 2024

@author: leehj
"""

# 샤프닝 예제 (c2_Sharpening.py)

# 관련 라이브러리 선언
import numpy as np
import cv2
from imgRead import imgRead
from createFolder import createFolder

# 영상 읽기
img = imgRead("../images/img9.jpg", cv2.IMREAD_GRAYSCALE, 320, 240)

middle = int(img.shape[0] / 2)
center = int(img.shape[1] / 2)

for y in range(img.shape[0]):
    img[y, center] = 255

for x in range(img.shape[1]):
    img[middle, x] = 255
    
cv2.imshow("lion", img)

# 키보드 입력을 기다린 후 모든 영상창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()
