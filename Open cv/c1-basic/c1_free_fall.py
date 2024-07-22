# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:37:03 2024

@author: leehj
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 13 17:32:33 2024

@author: Solero
"""

# 자유 낙하 운동

import cv2
from math import *
import numpy as np

cv2.namedWindow("Free Fall")

width = 512
height = 960

img = np.zeros((height, width, 3), np.uint8)

time = ypos = 0

help(cv2.circle)
while (True):
    if (ypos + 30) < height:
        cv2.circle(img, (256, 30+ypos), 10, (255, 0, 0), -1)
        time += 1
        ypos = int((9.8 * time ** 2) / 2)
        print(time, ':', ypos)
        
    cv2.imshow('Clock', img)
    if cv2.waitKey(100) >= 0:
        break
    

cv2.destroyWindow("Free Fall")    