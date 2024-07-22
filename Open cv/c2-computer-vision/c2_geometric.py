# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:07:50 2024

@author: leehj
"""

# 이동 변환, 크기 변환 예제 (c2_geometric.py)

# 관련 라이브러리 선언
import numpy as np
import cv2
from imgRead import imgRead
from createFolder import createFolder

# 영상 읽기
img1 = imgRead("../images/img11.jpg", cv2.IMREAD_GRAYSCALE, 320, 240)

# 이동 변환
h, w, = img1.shape
tlans_x = 10; tlans_y = 25
point1_src = np.float32([[15,20], [50,70], [130,140]])   # 소스 위치
point1_dst = np.float32(point1_src + [tlans_x, tlans_y]) # 타겟 위치
affine_mat1 = cv2.getAffineTransform(point1_src, point1_dst)
user_mat1 = np.float32([[1,0,tlans_x], [0,1,tlans_y]])
res1 = cv2.warpAffine(img1, affine_mat1, (w,h))
res2 = cv2.warpAffine(img1, user_mat1, (w,h))

# 크기 변환
scale_x = 0.8; scale_y = 0.6
background = np.full(shape=[h,w], fill_value=0, dtype=np.uint8)
user_mat2 = np.float32([[scale_x,0,0], [0,scale_y,0]])
res3 = cv2.warpAffine(img1, user_mat2, (w,h))
res4 = cv2.resize(img1, (0,0), None, scale_x, scale_y)
background[:(int)(h*scale_y), :(int)(w*scale_x)] = res4; res4 = background

# 이동 및 크기 변환
# 크기: x = 0.4, y = 0.6
# 이동: x = 100, y = 50
user_mat3 = np.float32([[0.4, 0, 100], [0, 0.6, 50]])
res5 = cv2.warpAffine(img1, user_mat3, (w,h))

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
save_dir = '../code_res_imgs/c2_geometric'
createFolder(save_dir)
for (name, out) in displays:
    cv2.imwrite(save_dir + "/" + name + ".jpg", out)