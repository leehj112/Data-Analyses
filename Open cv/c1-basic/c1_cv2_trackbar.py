# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:36:21 2024

@author: leehj
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:37:21 2024

@author: Solero
"""

import cv2
import numpy as np

ESC = 27

def nothing(pos):
    print(f"nothing: pos({pos})")

cv2.namedWindow("RGB track bar")
cv2.createTrackbar('Red color', 'RGB track bar', 0, 255, nothing)
cv2.createTrackbar('Green color', 'RGB track bar', 0, 255, nothing)
cv2.createTrackbar('Blue color', 'RGB track bar', 0, 255, nothing)

cv2.setTrackbarPos("Red color", 'RGB track bar', 125)
cv2.setTrackbarPos("Green color", 'RGB track bar', 125)
cv2.setTrackbarPos("Blue color", 'RGB track bar', 125)

img = np.zeros((512, 512, 3), np.uint8)
    
while(1):
    redval = cv2.getTrackbarPos("Red color", 'RGB track bar')
    greenval = cv2.getTrackbarPos("Green color", 'RGB track bar')
    blueval = cv2.getTrackbarPos("Blue color", 'RGB track bar')
    
    cv2.rectangle(img, (0,0), (512,512), (blueval, greenval, redval), -1)    
    cv2.imshow('RGB track bar', img)
    
    if cv2.waitKey(1) & 0xFF == ESC:
        break
    
cv2.destroyAllWindows()    
    