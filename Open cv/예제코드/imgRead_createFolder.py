# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:35:30 2024

@author: leehj
"""

import os

# 폴더 생성하기
def createFolder(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print ('Error: createFolder')
        
#%%
import cv2

# 영상을 불러오기
def imgRead(imgPath, imgReadType, imgResWidth, imgResHeight):
    img = cv2.imread(imgPath, imgReadType)
    if imgResHeight != 0 | imgResWidth != 0:
        img = cv2.resize(img, (imgResWidth, imgResHeight))
    return img