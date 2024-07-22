# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:14:04 2024

@author: leehj
"""

import cv2

# 영상을 불러오기
def imgRead(imgPath, imgReadType, imgResWidth, imgResHeight):
    img = cv2.imread(imgPath, imgReadType)
    if imgResHeight != 0 | imgResWidth != 0:
        img = cv2.resize(img, (imgResWidth, imgResHeight))
    return img