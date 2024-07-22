# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:28:26 2024

@author: leehj
"""

# 마우스 이벤트 활용
# P99

import cv2
import numpy as np

def draw_rectangle(event, x, y, flags, param):
    # print(f"x({x}), y({y})")
    if event == cv2.EVENT_LBUTTONDBLCLK: # 왼쪽 마우스 더블 클릭
        # 사각형 도형 출력
        cv2.rectangle(img, (x,y), (x+50, y+50), (0,0,255), -1) # BGR: Red
        
img = np.zeros((512,512,3), np.uint8)

cv2.namedWindow('image')

# 마우스 이벤트 리스너 등록
cv2.setMouseCallback('image', draw_rectangle) 

while(1):
    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == 27: # ESC
        break
    
cv2.destroyAllWindows()    