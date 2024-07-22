# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:11:25 2024

@author: leehj
"""

# 열기, 닫기 예제 (c2_morphology.py)

# 관련 라이브러리 선언
import numpy as np
import cv2
from imgRead import imgRead
from createFolder import createFolder

# 영상 읽기
img1 = imgRead("../images/img9.jpg", cv2.IMREAD_GRAYSCALE, 320, 240)

# 모톨로지 연산 수행
methods = [cv2.MORPH_OPEN,
           cv2.MORPH_CLOSE,
           cv2.MORPH_GRADIENT,
           cv2.MORPH_TOPHAT,
           cv2.MORPH_BLACKHAT,
           cv2.MORPH_HITMISS]
ress = []
for method in methods:
    res = cv2.morphologyEx(img1, method, cv2.UMat(), iterations=1)
    ress.append(res)

# 결과 영상 출력
displays = [("input1", img1),
            ("res1", ress[0]),
            ("res2", ress[1]),
            ("res3", ress[2]),
            ("res4", ress[3]),
            ("res5", ress[4]),
            ("res6", ress[5])]
for (name, out) in displays:
    cv2.imshow(name, out)

# 키보드 입력을 기다린 후 모든 영상창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()

# 영상 저장
save_dir = "../code_res_imgs/c2_morphology"
createFolder(save_dir)
for (name, out) in displays:
    cv2.imwrite(save_dir + "/" + name + ".jpg", out)