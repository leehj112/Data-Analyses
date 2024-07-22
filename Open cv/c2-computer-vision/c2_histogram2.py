# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:09:04 2024

@author: leehj
"""

#히스토그램 계산 예제 (c2_histogram1.py)

# 관련 라이브러리 선언
import numpy as np
import cv2
from matplotlib import pyplot as plt
from imgRead import imgRead
from createFolder import createFolder

# 영상 읽기
img1 = imgRead("./images/img5.jpg", cv2.IMREAD_UNCHANGED, 320, 240)

# 히스토그램 계산
ch1 = [2]; ch2 = [2]; ch3 = [2]
# ch1 = [0]; ch2 = [0]; ch3 = [0]
# ch1 = [0]; ch2 = [0]; ch3 = [0]
ranges1 = [0, 256]; ranges2 = [0, 128]; ranges3 = [128, 256]
histSize1 = [256]; histSize2 = [128]; histSize3 = [128]

# 채널에서 특정 값 범위에 대하여 총 빈 개수 계산
hist1 = cv2.calcHist([img1], ch1, None, histSize1, ranges1)
hist2 = cv2.calcHist([img1], ch2, None, histSize2, ranges2)
hist3 = cv2.calcHist([img1], ch3, None, histSize3, ranges3)

# 히스토그램 출력 및 저장
bin_x1 = np.arange(256)
bin_x2 = np.arange(128)
bin_x3 = np.arange(128) + 128

plt.title("Histogram")
plt.xlabel("Bin")
plt.ylabel("Frequency")
plt.plot(bin_x1, hist1, color='b')             # 선차트
plt.bar(bin_x2, hist2[:,0], width=6,color='r') # 바차트 : 빨강
plt.bar(bin_x3, hist3[:,0], width=6,color='g') # 바차드 : 초록
plt.grid(True, lw = 1, ls = '--', c= '.75')
plt.xlim([0,255])

# 영상 저장
save_dir = './code_res_imgs/c2_histogram1'
createFolder(save_dir)
plt.savefig(save_dir + "/" + "hist.png")

plt.show()