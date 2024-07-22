# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:04:07 2024

@author: leehj
"""

# HOG, LBP 기반 사람, 얼굴 검출 예제 (c3_HOG_LBP_Detection.py)

# 관련 라이브러리 선언
import cv2
from imgRead import imgRead
from createFolder import createFolder

# 영상 읽기
img1 = imgRead("./images/img_6_6.jpg", cv2.IMREAD_GRAYSCALE, 320, 240)
img2 = imgRead("./images/img50.jpg", cv2.IMREAD_GRAYSCALE, 320, 240)

# HOG 기반 사람검지
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
(hog_peoples, weights) = hog.detectMultiScale(img1, winStride=(8,8), padding=(32,32), scale=1.05)

# LBP 기반 얼굴검지
lbp_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')
lbp_faces = lbp_cascade.detectMultiScale(img2, 1.5, 1)

# 결과 영상 출력
res1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
for (x, y, w, h) in hog_peoples:
    cv2.rectangle(res1, (x, y), (x + w, y + h), (255, 0, 0), 2)

res2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
for (x, y, w, h) in lbp_faces:
    cv2.rectangle(res2, (x, y), (x + w, y + h), (255, 0, 0), 2)

displays = [("input1", img1),
            ("res1", res1),
            ("input2", img2),
            ("res2", res2)]
for (name, out) in displays:
    cv2.imshow(name, out)

# 키보드 입력을 기다린 후 모든 영상창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()

# 영상 저장
save_dir = "./code_res_imgs/c3_HOG_LBP_Detection"
createFolder(save_dir)
for (name, out) in displays:
    cv2.imwrite(save_dir + "/" + name + ".jpg", out)