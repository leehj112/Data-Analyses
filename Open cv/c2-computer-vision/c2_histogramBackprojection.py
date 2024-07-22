# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:09:20 2024

@author: leehj
"""

# 히스토그램 역투영 예제 (c2_histogramBackprojection.py)
# 한하는 색상을 가지는 영역을 추출
# 유사한 컬러 특성을 가지는 영역을 찾을 때

# 관련 라이브러리 선언
import numpy as np
import cv2
from matplotlib import pyplot as plt
from imgRead import imgRead
from createFolder import createFolder

# 영상 읽기
img1 = imgRead("../images/img7.jpg", cv2.IMREAD_UNCHANGED, 320, 240)

# 컬러 공간 변환 및 채널 분할 
# 색상(H), 채도(S), 명도(V)
img1_HSV = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
img1_H, img1_S, img1_V = cv2.split(img1_HSV)

# 입력 히스토그램 생성
ch1 = [0]; ranges1 = [0, 256]; histSize1 = [256]; bin_x1 = np.arange(256)

# 행(시작:종료), 열(시작:종료)
mask1 = img1_H[110:150, 200:240]

hist_mask = cv2.calcHist([mask1], ch1, None, histSize1, ranges1)

# 히스토그램 역투영 수행
bp = cv2.calcBackProject([img1_H], ch1, hist_mask, ranges1, 1)

# 결과영상
# thres(임계값) : 0
# maxval(최대값): 255
# 0보다 크면 255, 0보다 작거나 같으면 0
ret1, res1 = cv2.threshold(bp, 0, 255, cv2.THRESH_BINARY)

mask2 = np.full(shape=img1.shape, fill_value=0, dtype=np.uint8)
mask2[:,:,0] = res1
mask2[:,:,1] = res1 
mask2[:,:,2] = res1
res2 = cv2.bitwise_and(img1, mask2)

# 사각형(rectangle): 시작점(left, top), 종료점(right, bottom)
# 컬러: white(흰색)
# 두께: 2픽셀
cv2.rectangle(img1, (200, 110), (240, 150), (255, 255, 255), 2)
cv2.imshow("img1+mask", img1)
cv2.imshow("res1(binary)", res1)
cv2.imshow("res1(overlap)", res2)

# 히스토그램 출력 및 결과 저장
save_dir = '../code_res_imgs/c4_3'
createFolder(save_dir)
cv2.imwrite(save_dir + "/" + "input1.png", img1)
cv2.imwrite(save_dir + "/" + "res1.png", res1)
cv2.imwrite(save_dir + "/" + "res2.png", res2)

hist_Hue = cv2.calcHist([img1_H], ch1, None, histSize1, ranges1)
hist_res = cv2.calcHist([bp], ch1, None, histSize1, ranges1)
displays = [("Input Histogram", hist_Hue, 0, np.arange(256), 1),
            ("Mask Histogram", hist_mask, 0, np.arange(256), 2),
            ("Res Histogram", hist_res, 1, np.arange(255), 3)]

for (name, out, hist_index, hist_bins, index) in displays:
    plt.figure(index)
    plt.title(name); plt.xlabel("Bin"); plt.ylabel("Frequency")
    plt.bar(hist_bins, out[hist_index:, 0], width=6, color='g')
    plt.grid(True, lw=1, ls='--', c='.75')
    plt.xlim([0, 255])
    plt.savefig(save_dir + "/" + name + ".png")

plt.show()

# 키보드 입력을 기다린 후 모든 영상창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()