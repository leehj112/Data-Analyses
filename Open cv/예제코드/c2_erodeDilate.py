# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 14:59:56 2024

@author: leehj
"""

# 침식 팽창 예제 (c2_erodeDilate.py)

# 관련 라이브러리 선언
import cv2
from imgRead import imgRead
from createFolder import createFolder

#트랙바 호출함수
def nothing(x):
    pass

def check_odd(num):
    if num % 2 == 0:
        num += 1
    return num

def set_run(pos):
    global img1
    global img_index
    method = cv2.getTrackbarPos('method', "morphology")
    itr = cv2.getTrackbarPos('iter', "morphology")
    ksize = cv2.getTrackbarPos('ksize', "morphology")
    run = cv2.getTrackbarPos('run', "morphology")
    if run == 1:
        ksize = check_odd(ksize)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        if method == 0:
            res = cv2.erode(img1, kernel, iterations=itr)
        else:
            res = cv2.dilate(img1, kernel, iterations=itr)
        cv2.imshow("morphology", res)
        cv2.imwrite(save_dir + "/res" + str(img_index) + ".jpg", res)
        img_index += 1

# 영상 읽기
img1 = imgRead('./images/img9.jpg', cv2.IMREAD_GRAYSCALE, 320, 240)

# 트랙바 생성
img_index = 1
save_dir = "./code_res_imgs/c2_erodeDilate"
createFolder(save_dir)

cv2.namedWindow('morphology')
cv2.createTrackbar('method', 'morphology', 0, 1, nothing)
cv2.createTrackbar('ksize', 'morphology', 3, 10, nothing)
cv2.createTrackbar("iter", 'morphology', 1, 10, nothing)
cv2.createTrackbar("run", 'morphology', 0, 1, set_run)
cv2.setTrackbarPos("method", "morphology", 0)
cv2.setTrackbarPos("ksize", "morphology", 3)
cv2.setTrackbarPos("iter", "morphology", 1)
cv2.setTrackbarPos("run", "morphology", 0)

cv2.imshow("morphology", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()