# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:49:20 2024

@author: leehj
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 14:32:12 2023

@author: Solero
"""

# 아날로그 시계

import cv2
import time
from math import *
import numpy as np

cv2.namedWindow('Clock')

img = np.zeros((512,512,3), np.uint8)


#%%

def draw_background():
    cv2.circle(img, (256, 256), 250, (125, 125, 125), -1)
    
    for hour in range(12):
        Ang_Hour = hour*30

        if (0<=hour<=2):
            x_pos = int(250.0*cos((90.0-Ang_Hour)*3.141592 / 180))
            y_pos = int(250.0*sin((90.0-Ang_Hour)*3.141592 / 180))
            cv2.circle(img, (256 + x_pos, 256 - y_pos), 5, (255, 125, 125), 3)
        elif(3<=hour<=5):
            x_pos = int(250.0*cos((Ang_Hour-90.0)*3.141592 / 180))
            y_pos = int(250.0*sin((Ang_Hour-90.0)*3.141592 / 180))
            cv2.circle(img, (256 + x_pos, 256 + y_pos), 5, (255, 125, 125), 3)
        elif(6<=hour<=8):
            if hour != 6:
                x_pos = int(250.0*cos((Ang_Hour-180.0)*3.141592 / 180))
                y_pos = int(250.0*sin((Ang_Hour-180.0)*3.141592 / 180))
            else:
                x_pos = 0
                y_pos = 250
            cv2.circle(img, (256 - x_pos, 256 + y_pos), 5, (255, 125, 125), 3)
        elif(9<=hour<=11):
            x_pos = int(250.0*cos((Ang_Hour-270.0)*3.141592 / 180))
            y_pos = int(250.0*sin((Ang_Hour-270.0)*3.141592 / 180))
            cv2.circle(img, (256 - x_pos, 256 - y_pos), 5, (255, 125, 125), 3)


#%%
while(True):
    draw_background()
    
    now = time.localtime()
    hour = now.tm_hour
    min = now.tm_min
    sec = now.tm_sec
    
    if hour > 12:
        hour -= 12

    Ang_Min = min * 6
    Ang_Hour = hour * 30 + min * 0.5
    Ang_Sec = sec * 6

    print(f"{hour}:{min}:{sec}: {Ang_Hour}:{Ang_Min}:{Ang_Sec}")

    # 시침
    if (hour ==12 or 1<=hour<=2):
        x_pos = int(150.0*cos((90.0-Ang_Hour)*3.141592 /180))
        y_pos = int(150.0*sin((90.0-Ang_Hour)*3.141592 /180))
        cv2.line(img, (256, 256), (256 + x_pos, 256 - y_pos), (255, 0, 0), 6)
    elif(3<=hour<=5):
        x_pos = int(150.0*cos((Ang_Hour-90.0)*3.141592 /180))
        y_pos = int(150.0*sin((Ang_Hour-90.0)*3.141592 /180))
        cv2.line(img, (256, 256), (256 + x_pos, 256 + y_pos), (255, 0, 0), 6)
    elif(6<=hour<=8):
        x_pos = int(150.0*cos((Ang_Hour-180.0)*3.141592 /180))
        y_pos = int(150.0*sin((Ang_Hour-180.0)*3.141592 /180))
        cv2.line(img, (256, 256), (256 - x_pos, 256 + y_pos), (255, 0, 0), 6)
    elif(9<=hour<=11):
        x_pos = int(150.0*cos((Ang_Hour-270.0)*3.141592 /180))
        y_pos = int(150.0*sin((Ang_Hour-270.0)*3.141592 /180))
        cv2.line(img, (256, 256), (256 - x_pos, 256 - y_pos), (255, 0, 0), 6)
    
    # 분침
    if(min == 0 or 1<=hour<=14):
        x_pos = int(210.0*cos((90-Ang_Min)*3.141592 /180))
        y_pos = int(210.0*sin((90-Ang_Min)*3.141592 /180))
        cv2.line(img, (256, 256), (256 + x_pos, 256 - y_pos), (0, 255, 0), 3)
    elif(15<= hour <= 29):
        x_pos = int(210.0*cos((Ang_Min-90)*3.141592 /180))
        y_pos = int(210.0*sin((Ang_Min-90)*3.141592 /180))
        cv2.line(img, (256, 256), (256 + x_pos, 256 + y_pos), (0, 255, 0), 3)
    elif(30<= hour <= 44):
        x_pos = int(210.0*cos((Ang_Min-180)*3.141592 /180))
        y_pos = int(210.0*sin((Ang_Min-180)*3.141592 /180))
        cv2.line(img, (256, 256), (256 - x_pos, 256 + y_pos), (0, 255, 0), 3)
    elif(45<= hour <= 59):
        x_pos = int(210.0*cos((Ang_Min-270)*3.141592 /180))
        y_pos = int(210.0*sin((Ang_Min-270)*3.141592 /180))
        cv2.line(img, (256, 256), (256 - x_pos, 256 - y_pos), (0, 255, 0), 3)
    
    # 초침
    if(sec == 0 or 1 <= hour <= 14):
        x_pos = int(230.0*cos((90-Ang_Sec)*3.141592 /180))
        y_pos = int(230.0*sin((90-Ang_Sec)*3.141592 /180))
        cv2.line(img, (256, 256), (256 + x_pos, 256 - y_pos), (0, 0, 255), 2)
    elif(15<= hour <= 29):
        x_pos = int(230.0*cos((Ang_Sec-90)*3.141592 /180))
        y_pos = int(230.0*sin((Ang_Sec-90)*3.141592 /180))
        cv2.line(img, (256, 256), (256 + x_pos, 256 + y_pos), (0, 0, 255), 2)
    elif(30<= hour <= 44):
        x_pos = int(230.0*cos((Ang_Sec-180)*3.141592 /180))
        y_pos = int(230.0*sin((Ang_Sec-180)*3.141592 /180))
        cv2.line(img, (256, 256), (256 - x_pos, 256 + y_pos), (0, 0, 255), 2)
    elif(45<= hour <= 59):
        x_pos = int(230.0*cos((Ang_Sec-270)*3.141592 /180))
        y_pos = int(230.0*sin((Ang_Sec-270)*3.141592 /180))
        cv2.line(img, (256, 256), (256 - x_pos, 256 - y_pos), (0, 0, 255), 2)

    cv2.imshow('Clock', img)
    if cv2.waitKey(1000) >= 0: # milli-seconds
        break

cv2.destroyAllWindows()