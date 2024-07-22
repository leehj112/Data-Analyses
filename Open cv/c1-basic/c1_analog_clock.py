# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:48:24 2024

@author: leehj
"""

# -*- coding: utf-8 -*-

# 아날로그 시계
"""
x = cos(s)
y = sin(s)
"""

import cv2
import time
from math import *
import numpy as np

cv2.namedWindow('Clock')

img = np.zeros((512,512,3), np.uint8)

while(True):
    cv2.circle(img, (256, 256), 250, (125, 125, 125), -1)

    now = time.localtime() # 현재시간
    hour = now.tm_hour
    min = now.tm_min
    
    if hour > 12:
        hour -= 12

    # 각도(angle)
    # 분: 60분 * 6 -> 360도
    # 시: 12시간 * 30 -> 360도, 시간당 30도 이동
    Ang_Min = min * 6
    Ang_Hour = hour*30 + min*0.5

    # 시침
    if (hour ==12 or 1<=hour<=2):
        x_pos = int(150.0*cos((90.0-Ang_Hour)*3.141592 /180))
        y_pos = int(150.0*sin((90.0-Ang_Hour)*3.141592 /180))
        cv2.line(img, (256, 256), (256 + x_pos, 256 - y_pos), (0, 255, 0), 6)
    elif(3<=hour<=5):
        x_pos = int(150.0*cos((Ang_Hour-90.0)*3.141592 /180))
        y_pos = int(150.0*sin((Ang_Hour-90.0)*3.141592 /180))
        cv2.line(img, (256, 256), (256 + x_pos, 256 + y_pos), (0, 255, 0), 6)
    elif(6<=hour<=8):
        x_pos = int(150.0*cos((Ang_Hour-180.0)*3.141592 /180))
        y_pos = int(150.0*sin((Ang_Hour-180.0)*3.141592 /180))
        cv2.line(img, (256, 256), (256 - x_pos, 256 + y_pos), (0, 255, 0), 6)
    elif(9<=hour<=11):
        x_pos = int(150.0*cos((Ang_Hour-270.0)*3.141592 /180))
        y_pos = int(150.0*sin((Ang_Hour-270.0)*3.141592 /180))
        cv2.line(img, (256, 256), (256 - x_pos, 256 - y_pos), (0, 255, 0), 6)

    # 분침
    if(min == 0 or 1<=hour<=14):
        x_pos = int(200.0*cos((90-Ang_Min)*3.141592 /180))
        y_pos = int(200.0*sin((90-Ang_Min)*3.141592 /180))
        cv2.line(img, (256, 256), (256 + x_pos, 256 - y_pos), (0, 255, 0), 1)
    elif(15<= hour <= 29):
        x_pos = int(200.0*cos((Ang_Min-90)*3.141592 /180))
        y_pos = int(200.0*sin((Ang_Min-90)*3.141592 /180))
        cv2.line(img, (256, 256), (256 + x_pos, 256 + y_pos), (0, 255, 0), 1)
    elif(30<= hour <= 44):
        x_pos = int(200.0*cos((Ang_Min-180)*3.141592 /180))
        y_pos = int(200.0*sin((Ang_Min-180)*3.141592 /180))
        cv2.line(img, (256, 256), (256 - x_pos, 256 + y_pos), (0, 255, 0), 1)
    elif(45<= hour <= 59):
        x_pos = int(200.0*cos((Ang_Min-270)*3.141592 /180))
        y_pos = int(200.0*sin((Ang_Min-270)*3.141592 /180))
        cv2.line(img, (256, 256), (256 - x_pos, 256 - y_pos), (0, 255, 0), 1)

    # 초침
    cv2.imshow('Clock', img)
    if cv2.waitKey(10)>=0:
        break

cv2.destroyAllWindows()