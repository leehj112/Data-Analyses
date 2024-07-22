# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 14:58:55 2024

@author: leehj
"""

#영상 명암비 조절 예제 (c2_contrast.py)

# 관련 라이브러리 선언
import numpy as np
import cv2
from imgRead import imgRead
from createFolder import createFolder

# 영상 읽기
img1 = imgRead("./images/img5.jpg", cv2.IMREAD_GRAYSCALE, 320, 240)

# 영상 명암비 조절 변수 선언 및 초기화
multi_lut = np.full(shape=[256], fill_value=0, dtype=np.uint8)
log_lut = np.full(shape=[256], fill_value=0, dtype=np.uint8)
invol_lut = np.full(shape=[256], fill_value=0, dtype=np.uint8)
sel_lut = np.full(shape=[256], fill_value=0, dtype=np.uint8)
multi_v = 2; gamma1 = 0.1; gamma2= 0.6
thres1 = 5; thres2 = 100
max_v_log = 255 / np.log(1 + 255)
max_v_invol = 255 / np.power(255, gamma1)
max_v_sel = 100 / np.power(thres2, gamma2)

for i in range(256):
    val = i * multi_v
    if val > 255 : val = 255
    multi_lut[i] = val
    log_lut[i] = np.round(max_v_log * np.log(1+i))
    invol_lut[i] = np.round(max_v_invol * np.power(i, gamma1))
    if i < thres1 : sel_lut[i] = i
    elif i > thres2 : sel_lut[i] = i
    else: sel_lut[i] =  np.round(max_v_sel * np.power(i, gamma2))

# 명암비 조절
ress = []
ress.append(cv2.LUT(img1, multi_lut))
ress.append(cv2.LUT(img1, log_lut))
ress.append(cv2.LUT(img1, invol_lut))
ress.append(cv2.LUT(img1, sel_lut))

# 결과 영상 출력
displays = [("input1", img1),
            ("res1", ress[0]),
            ("res2", ress[1]),
            ("res3", ress[2]),
            ("res4", ress[3])]
for (name, out) in displays:
    cv2.imshow(name, out)

# 키보드 입력을 기다린 후 모든 영상창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()

# 영상 저장
save_dir = './code_res_imgs/c2_contrast'
createFolder(save_dir)
for (name, out) in displays:
    cv2.imwrite(save_dir + "/" + name + ".jpg", out)