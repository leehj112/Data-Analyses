# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:37:22 2024

@author: leehj
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 14 11:14:46 2024

@author: Solero
"""

# 포물선 운동

import cv2
from math import *
import numpy as np


init_g = 9.8    # 중력가속도: 9.8 m/s²
init_vel = 80.0 # 초기속도
init_ang = 45.0 # 초기각도

cv2.namedWindow("Parabolic Motion")

width = 1024
height = 960

img = np.zeros((height, width, 3), np.uint8)

time = xpos = ypos = 0

init_posx = 30
init_posy = 250

vel_x = int(init_vel * cos(init_ang * pi / 180.0))
vel_y = int(-1.0 * init_vel * sin(init_ang * pi / 180.0))

while (True):
    if (ypos+30) < height:
        cv2.circle(img, (init_posx + xpos, init_posy + ypos), 10, (255,0,0), -1)
        time += 0.2
        
        vel_y = int(vel_y + init_g * time)
        
        xpos = int(xpos + vel_x * time)
        ypos = ypos + int(vel_y * time) + int((init_g * time**2) / 2)
        print(time, ':', xpos, ypos)
        
    cv2.imshow("Parabolic Motion", img)
    
    if cv2.waitKey(100) >= 0:
        break
        
        
cv2.destroyWindow("Parabolic Motion")        
        