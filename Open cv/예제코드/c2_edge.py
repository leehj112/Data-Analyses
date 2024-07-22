# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 14:59:36 2024

@author: leehj
"""

#에지 검지 예제 (c2_edge.py)

# 관련 라이브러리 선언
import numpy as np
import cv2
from imgRead import imgRead
from createFolder import createFolder

# 영상 읽기
img1 = imgRead("./images/img16.jpg", cv2.IMREAD_GRAYSCALE, 320, 240)

# 에지 추출
img1_blue = cv2.GaussianBlur(img1, (3,3), 0)
res1 = cv2.Sobel(img1, cv2.FILTER_SCHARR, 1, 0, ksize=3)
res2 = cv2.Scharr(img1_blue, cv2.CV_32FC1, 0, 1)
res3 = cv2.Laplacian(img1_blue, cv2.CV_32FC1)
res4 = cv2.Canny(img1, 50, 200, apertureSize=5, L2gradient=True)

# 결과 영상 출력
displays = [("input1", img1),
            ("input2", img1_blue),
            ("res1", res1),
            ("res2", res2),
            ("res3", res3),
            ("res4", res4)]
for (name, out) in displays:
    cv2.imshow(name, out)

# 키보드 입력을 기다린 후 모든 영상창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()

# 영상 저장
save_dir = "./code_res_imgs/c2_edge"
createFolder(save_dir)
for (name, out) in displays:
    cv2.imwrite(save_dir + "/" + name + ".jpg", out)