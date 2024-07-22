# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:06:39 2024

@author: leehj
"""

# 영상 군집화 예제 (c3_knn.py)

# 관련 라이브러리 선언
import numpy as np
import cv2
from imgRead import imgRead
from createFolder import createFolder

save_dir = './code_res_imgs/c3_knn'
createFolder(save_dir)

# 영상 읽기
img1 = imgRead("./images/img25.png", cv2.IMREAD_UNCHANGED, 320, 240)
cv2.imshow("input",img1)
cv2.imwrite(save_dir + "/input.jpg", img1)

# knn 입력 생성
data = img1.reshape((-1,3))
data = np.float32(data)

# knn 변수 설정
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
attempts = 10
flags = cv2.KMEANS_RANDOM_CENTERS
for i in range(1, 5):
    # knn 수행
    numK = i * 5
    ret, label, center = cv2.kmeans(data, numK, None, criteria, attempts, flags)

    # 결과 영상 출력 및 저장
    center = np.uint8(center)
    res = center[label.flatten()]
    res = res.reshape((img1.shape))
    cv2.imshow(str(numK),res)
    cv2.waitKey(1)
    cv2.imwrite(save_dir + "/" + str(numK) + "_knn.jpg", res)

# 키보드 입력을 기다린 후 모든 영상창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()