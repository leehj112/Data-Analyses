# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:01:58 2024

@author: leehj
"""

# 허프 변환 기반으로 직선 및 원 검출 예제 (c2_houghTransform.py)

# 관련 라이브러리 선언
import numpy as np
import cv2
from imgRead import imgRead
from matplotlib import pyplot as plt
from createFolder import createFolder

# 영상 읽기
img1 = imgRead("./images/img_6_3.png", cv2.IMREAD_GRAYSCALE, 320, 240)
img2 = imgRead("./images/img_6_0.png", cv2.IMREAD_GRAYSCALE, 320, 240)

# 직선 검출
img1_edge = cv2.Canny(img1, 50, 150, apertureSize=3)
lines = cv2.HoughLines(img1_edge, 2, np.pi/180, 100)
linesP = cv2.HoughLinesP(img1_edge, 2, np.pi/180, 50, minLineLength=1, maxLineGap=100)

# 원 검출
circles = cv2.HoughCircles(img2, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=20,
                           minRadius=30, maxRadius=50)

# 결과 출력
img1_color1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
if lines.any() != None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta); b=np.sin(theta)
        x0 = a*rho; y0 = b*rho
        x1 = int(x0 + 1000*(-b)); y1 = int(y0 + 1000*a)
        x2 = int(x0 - 1000*(-b)); y2 = int(y0 - 1000*a)
        cv2.line(img1_color1, (x1,y1), (x2,y2), (0,0,255), 2)

img1_color2 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
if linesP.any() != None:
    for line in linesP:
        x1, y1, x2, y2 = line[0]
        cv2.line(img1_color2, (x1,y1), (x2,y2), (0,255,0), 2)

circles = np.uint16(np.around(circles))
img2_color1 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
if circles.any() != None:
    for i in circles[0, :]:
        cv2.circle(img2_color1, (i[0], i[1]), i[2], (0, 0, 255), 2)

# 결과 영상 출력
displays = [("input1", img1),
            ("input2", img2),
            ("res1", img1_color1),
            ("res2", img1_color2),
            ("res3", img2_color1)]
for (name, out) in displays:
    cv2.imshow(name, out)

# 키보드 입력을 기다린 후 모든 영상창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()

# 영상 저장
save_dir = './code_res_imgs/c2_houghTransform'
createFolder(save_dir)
for (name, out) in displays:
    cv2.imwrite(save_dir + "/" + name + ".jpg", out)