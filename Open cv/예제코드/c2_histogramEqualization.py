# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:01:37 2024

@author: leehj
"""

# 히스토그램 평활화 예제 (c2_histogramEqualization.py)

# 관련 라이브러리 선언
import numpy as np
import cv2
from matplotlib import pyplot as plt
from imgRead import imgRead
from createFolder import createFolder

# 영상 읽기
img1 = imgRead("./images/img6.jpg", cv2.IMREAD_GRAYSCALE, 320, 240)

#히스토그램 평활화 및 히스토그램 계산
res1 = cv2.equalizeHist(img1)
ch1 = [0]; ranges1 = [0, 256]; histSize1 = [256]
hist1 = cv2.calcHist([img1], ch1, None, histSize1, ranges1)
hist2 = cv2.calcHist([res1], ch1, None, histSize1, ranges1)

#상수곱, 로그곱, 거급제곱 변환 기반 명암비 조절 및 히스토그램 계산
multi_lut = np.full(shape=[256], fill_value=0, dtype=np.uint8)
log_lut = np.full(shape=[256], fill_value=0, dtype=np.uint8)
invol1_lut = np.full(shape=[256], fill_value=0, dtype=np.uint8)
multi_v = 2; gamma1 = 0.4
thres1 = 5; thres2 = 100
max_v_log = 255 / np.log(1 + 255)
max_v_invol1 = 255 / np.power(255, gamma1)

for i in range(256):
    val = i * multi_v
    if val > 255 : val = 255
    multi_lut[i] = val
    log_lut[i] = np.round(max_v_log * np.log(1+i))
    invol1_lut[i] = np.round(max_v_invol1 * np.power(i, gamma1))

# 명암비 조절
res2 = cv2.LUT(img1, multi_lut)
res3 = cv2.LUT(img1, log_lut)
res4 = cv2.LUT(img1, invol1_lut)

hist3 = cv2.calcHist([res2], ch1, None, histSize1, ranges1)
hist4 = cv2.calcHist([res3], ch1, None, histSize1, ranges1)
hist5 = cv2.calcHist([res4], ch1, None, histSize1, ranges1)

# 히스토그램 출력 및 결과 저장
bin_x = np.arange(256)
fig_index = 0
save_dir = './code_res_imgs/c2_histogramEqualization'
createFolder(save_dir)
display_img = [("input1", img1),
            ("res1", res1),
            ("res2", res2),
            ("res3", res3),
            ("res4", res4)]
for (name, out) in display_img:
    cv2.imshow(name, out)
    cv2.imwrite(save_dir + "/" + name + ".jpg", out)
dlsplay_hist = [("Input Histogram", hist1),
                ("Equalization-convert Histogram", hist2),
                ("Multiply-convert Histogram", hist3),
                ("log-convert Histogram", hist4),
                ("Invil-convert Histogram", hist5)]
for (name, out) in dlsplay_hist:
    plt.figure(fig_index)
    plt.title(name); plt.xlabel("Bin"); plt.ylabel("Frequency")
    plt.bar(bin_x, out[:,0], width=6, color='g')
    plt.grid(True, lw=1, ls='--', c='.75')
    plt.xlim([0, 255])
    plt.savefig(save_dir + "/" + name + ".png")
    fig_index += 1
plt.show()

# 키보드 입력을 기다린 후 모든 영상창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()