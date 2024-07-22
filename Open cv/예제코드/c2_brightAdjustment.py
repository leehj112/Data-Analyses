# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 14:58:40 2024

@author: leehj
"""

# 밝기 조절 예제 (c2_brightadjustment.py)

# 관련 라이브러리 선언
import numpy as np
import cv2
from imgRead import imgRead
from createFolder import createFolder

# 영상 읽기
img1 = imgRead("./images/img5.jpg", cv2.IMREAD_GRAYSCALE, 320, 240)

# 영상 밝기 조절 변수 선언 및 초기화
v = np.full(shape=img1.shape, fill_value=100, dtype=np.uint8)
v_n = np.full(shape=img1.shape, fill_value=255, dtype=np.uint8)

# 영상 밝기 조절
# res1은 파이썬 결과, res2과 res3는 OpenCV에서 제공하는 함수를 사용한 결과
ress = []
ress.append(np.uint8(img1 + v))
ress.append(cv2.add(img1, v))
ress.append(cv2.subtract(v_n, img1))

# 결과 영상 출력
displays = [("input1", img1),
            ("res1", ress[0]),
            ("res2", ress[1]),
            ("res3", ress[2])]
for (name, out) in displays:
    cv2.imshow(name, out)

# 키보드 입력을 기다린 후 모든 영상창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()

# 영상 저장
save_dir = './code_res_imgs/c2_brightadjustment'
createFolder(save_dir)
for (name, out) in displays:
    cv2.imwrite(save_dir+"/"+name+".jpg", out)