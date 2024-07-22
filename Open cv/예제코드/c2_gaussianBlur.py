# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:00:16 2024

@author: leehj
"""

# 가우시안 블러링 예제 (c2_gaussianBlur.py)

# 관련 라이브러리 선언
import numpy as np
import cv2
from imgRead import imgRead
from createFolder import createFolder

# 영상 읽기
img1 = imgRead("./images/img8.jpg", cv2.IMREAD_GRAYSCALE, 320, 240)

# 필터 정의 및 블러링
ksize1 = 7; ksize2 = 9
res1 = cv2.GaussianBlur(img1, (ksize1,ksize1), 0)
res2 = cv2.GaussianBlur(img1, (ksize2,ksize2), 0)
res3 = cv2.GaussianBlur(img1, (1,21), 0)

# 결과 영상 출력
displays = [("input1", img1),
            ("res1", res1),
            ("res2", res2),
            ("res3", res3)]
for (name, out) in displays:
    cv2.imshow(name, out)

# 키보드 입력을 기다린 후 모든 영상창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()

# 영상 저장
save_dir = './code_res_imgs/c2_gaussianBlur'
createFolder(save_dir)
for (name, out) in displays:
    cv2.imwrite(save_dir+"/"+name+".jpg", out)