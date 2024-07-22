# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:35:56 2024

@author: leehj
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 16:03:08 2023

@author: Solero
"""

# P95

import cv2 
import numpy as np

print(cv2.__version__) # 4.9.0

#%%
img = cv2.imread("./images/img_6_0.png") # 그림 읽기

print(img.shape) # (높이, 너비, 칼라) -> (493, 640, 3) 

#%%
height = img.shape[0] # 높이
width = img.shape[1]  # 너비
bgr = img.shape[2]    # 컬러(RGB)

print("height:", height)
print("width:", width)
print("bgr:", bgr)

#%%


# 세로선
for y in range(0, height):
    # img.itemset(y, x, channel, color) : 해당 위치에 픽셀을 지정
    img.itemset(y, int(width / 2), 0, 0)   # blue
    img.itemset(y, int(width / 2), 1, 0)   # green
    img.itemset(y, int(width / 2), 2, 255) # red
    
# 가로선    
for x in range(0, width):    
    img.itemset(int(height / 2), x, 0, 255) # blue
    img.itemset(int(height / 2), x, 1, 0)   # green
    img.itemset(int(height / 2), x, 2, 0)   # red

   
#%%

cv2.namedWindow("image") # 이름을 가진 윈도우 생성
cv2.imshow("image", img)

cv2.waitKey()

cv2.destroyAllWindows() # 열려있는 모두 윈도우 닫음