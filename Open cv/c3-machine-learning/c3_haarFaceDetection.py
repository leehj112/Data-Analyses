# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:29:54 2024

@author: leehj
"""

# 얼굴 검출 예제 (c3_haarFaceDetection.py)

# https://github.com/opencv/opencv/tree/master/data/haarcascades

# 관련 라이브러리 선언
import cv2
from imgRead import imgRead
from createFolder import createFolder

# 영상 읽기
img1 = imgRead("../images/img_6_6.jpg", cv2.IMREAD_GRAYSCALE, 320, 240)

# 얼굴검지
face_cascade = cv2.CascadeClassifier('../models/haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(img1, 1.1, 3)

# 결과 영상 출력
res1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
for (x, y, w, h) in faces:
    cv2.rectangle(res1, (x, y), (x+w, y+h), (255, 0, 0), 2)
displays = [("input1", img1),
            ("res1", res1)]
for (name, out) in displays:
    cv2.imshow(name, out)

# 키보드 입력을 기다린 후 모든 영상창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()

# 영상 저장
save_dir = "../code_res_imgs/c3_faceDetection"
createFolder(save_dir)
for (name, out) in displays:
    cv2.imwrite(save_dir + "/" + name + ".jpg", out)