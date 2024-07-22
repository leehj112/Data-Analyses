# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:48:55 2024

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

for hour in range(3,6):
    for min in range(0,60):
        if hour > 12:
            hour -= 12

        Ang_Min = min * 6
        Ang_Hour = hour*30+min*0.5

        print(f"{hour}:{min}: {Ang_Hour}:{Ang_Min}")

        # 시침
        if (hour == 12 or 1<=hour<=2):
            x_pos = int(150.0*cos((90.0-Ang_Hour)*3.141592 /180))
            y_pos = int(150.0*sin((90.0-Ang_Hour)*3.141592 /180))
            print(f'시침: {x_pos}, {y_pos}')
        elif(3<=hour<=5):
            x_pos = int(150.0*cos((Ang_Hour-90.0)*3.141592 /180))
            y_pos = int(150.0*sin((Ang_Hour-90.0)*3.141592 /180))
            print(f'시침: {x_pos}, {y_pos}')
        elif(6<=hour<=8):
            x_pos = int(150.0*cos((Ang_Hour-180.0)*3.141592 /180))
            y_pos = int(150.0*sin((Ang_Hour-180.0)*3.141592 /180))
            print(f'시침: {x_pos}, {y_pos}')
        elif(9<=hour<=11):
            x_pos = int(150.0*cos((Ang_Hour-270.0)*3.141592 /180))
            y_pos = int(150.0*sin((Ang_Hour-270.0)*3.141592 /180))
            print(f'시침: {x_pos}, {y_pos}')
        
        # 분침
        if(min == 0 or 1<=hour<=14):
            x_pos = int(200.0*cos((90-Ang_Min)*3.141592 /180))
            y_pos = int(200.0*sin((90-Ang_Min)*3.141592 /180))
            print(f'분침: {x_pos}, {y_pos}')
        elif(15<= hour <= 29):
            x_pos = int(200.0*cos((Ang_Min-90)*3.141592 /180))
            y_pos = int(200.0*sin((Ang_Min-90)*3.141592 /180))
            print(f'분침: {x_pos}, {y_pos}')
        elif(30<= hour <= 44):
            x_pos = int(200.0*cos((Ang_Min-180)*3.141592 /180))
            y_pos = int(200.0*sin((Ang_Min-180)*3.141592 /180))
            print(f'분침: {x_pos}, {y_pos}')
        elif(45<= hour <= 59):
            x_pos = int(200.0*cos((Ang_Min-270)*3.141592 /180))
            y_pos = int(200.0*sin((Ang_Min-270)*3.141592 /180))
            print(f'분침: {x_pos}, {y_pos}')