# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 14:58:08 2024

@author: leehj
"""

# 산술 및 논리 연산 예제 (c2_arithmeticLogical.py)

# 관련 라이브러리 선언
import numpy as np
import cv2
from imgRead import imgRead
from createFolder import createFolder

# 영상 읽기
img1 = imgRead("./images/img1.jpg", cv2.IMREAD_UNCHANGED, 320, 240)
img2 = imgRead("./images/img2.jpg", cv2.IMREAD_UNCHANGED, 320, 240)
img3 = imgRead("./images/img3.jpg", cv2.IMREAD_UNCHANGED, 320, 240)
img4 = imgRead("./images/img4.jpg", cv2.IMREAD_UNCHANGED, 320, 240)
img5 = imgRead("./images/img5.jpg", cv2.IMREAD_GRAYSCALE, 320, 240)

# 마스크 선언 및 초기화
mask = np.full(shape=img5.shape, fill_value=0, dtype=np.uint8)
h, w = img5.shape
x = (int)(w/2) - 60; y = (int)(h/2) - 60
cv2.rectangle(mask, (x,y), (x+120, y+120), (255,255,255), -1)

# 산술 및 논리 연산 수행
ress = []
ress.append(cv2.add(img1, img2))
ress.append(cv2.addWeighted(img1, 0.5, img2, 0.5, 0))
ress.append(cv2.subtract(img3, img4))
ress.append(cv2.absdiff(img3, img4))
ress.append(cv2.bitwise_not(img5))
ress.append(cv2.bitwise_and(img5, mask))

# 결과 영상 출력
displays = [("input1", img1),
            ("input2", img2),
            ("input3", img3),
            ("input4", img4),
            ("input5", img5),
            ("res1", ress[0]),
            ("res2", ress[1]),
            ("res3", ress[2]),
            ("res4", ress[3]),
            ("res5", ress[4]),
            ("res6", ress[5]),]
for (name, out) in displays:
    cv2.imshow(name, out)

# 키보드 입력을 기다린 후 모든 영상창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()

# 영상 저장
save_dir = './code_res_imgs/c2_arithmeticLogical'
createFolder(save_dir)
for (name, out) in displays:
    cv2.imwrite(save_dir+"/"+name+".jpg", out)

# cv2.rectangle(img1, (10,10), (100,100), (255,255,255), 1, cv2.LINE_AA, cv2.LINE)
