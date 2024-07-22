# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:36:40 2024

@author: leehj
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 16:52:30 2023

@author: Solero
"""

# 동영상 파일

import cv2

# cap = cv2.VideoCapture("./images/video2.mp4")

# cap = cv2.VideoCapture("vtest.avi")
cap = cv2.VideoCapture(0) # Default camera

while cap.isOpened():
    success, frame = cap.read()
    if success:
        cv2.imshow('image', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if(key == 27): # Escape Keyboard
            break
    else:
        break
        
cap.release() # 카메라 해제

cv2.destroyAllWindows() # 창 닫기
        