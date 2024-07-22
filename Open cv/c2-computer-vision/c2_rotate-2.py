# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:13:05 2024

@author: leehj
"""

# 회전 변환 예제 (c2_rotate.py)

# 관련 라이브러리 선언
import numpy as np
import cv2
from imgRead import imgRead
from createFolder import createFolder

# 영상 읽기
img1 = imgRead("../images/img11.jpg", cv2.IMREAD_GRAYSCALE, 320, 240)

# 영상 회전
h, w, = img1.shape
c_h = 0
c_w = 0
rot_mat1 = cv2.getRotationMatrix2D((c_w, c_h), 45, 1)
rot_mat2 = cv2.getRotationMatrix2D((c_w, c_h), -45, 1)
res1 = cv2.warpAffine(img1, rot_mat1, (w,h))
res2 = cv2.warpAffine(img1, rot_mat2, (w,h))
res3 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
res4 = cv2.flip(img1, 1)
res5 = cv2.flip(img1, -1)

# 결과 영상 출력
displays = [("input1", img1),
            ("res1", res1),
            ("res2", res2),
            ("res3", res3),
            ("res4", res4),
            ("res5", res5)]
for (name, out) in displays:
    cv2.imshow(name, out)

# 키보드 입력을 기다린 후 모든 영상창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()

# 영상 저장
save_dir = '../code_res_imgs/c2_rotate'
createFolder(save_dir)
for (name, out) in displays:
    cv2.imwrite(save_dir + "/" + name + ".jpg", out)