# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:12:39 2024

@author: leehj
"""

# 투시 변환 예제 (c2_perspective.py)

# 관련 라이브러리 선언
import numpy as np
import cv2
from imgRead import imgRead
from createFolder import createFolder

# 영상 읽기
img1 = imgRead("../images/img12.jpg", cv2.IMREAD_GRAYSCALE, 320, 240)

# 투시 변환 수행
h, w = img1.shape
point1_src = np.float32([[1,1], [w-10,10], [5,h-5], [w-4,h-4]])
point1_dst = np.float32([[15,15], [w-60,15], [10,h-25], [w-100,h-50]])
point2_src = np.float32([[148,145], [168,144], [136,223], [188,222]])
point2_dst = np.float32([[136,145], [188,144], [136,223], [188,222]])
per_mat1 = cv2.getPerspectiveTransform(point1_src, point1_dst)
per_mat2 = cv2.getPerspectiveTransform(point2_src, point2_dst)
res1 = cv2.warpPerspective(img1, per_mat1, (w,h))
res2 = cv2.warpPerspective(img1, per_mat2, (w,h))

# 결과 영상 출력
displays = [("input1", img1),
            ("res1", res1),
            ("res2", res2)]
for (name, out) in displays:
    cv2.imshow(name, out)

# 키보드 입력을 기다린 후 모든 영상창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()

# 영상 저장
save_dir = "../code_res_imgs/c2_perspective"
createFolder(save_dir)
for (name, out) in displays:
    cv2.imwrite(save_dir + "/" + name + ".jpg", out)