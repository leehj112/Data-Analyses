# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:27:01 2024

@author: leehj
"""

# HOG, LBP 기반 사람, 얼굴 검출 예제 (c3_HOG_LBP_Detection.py)

# 관련 라이브러리 선언
import cv2
from imgRead import imgRead
from createFolder import createFolder

# 영상 읽기
img2 = imgRead("../images/peoples.jpg", cv2.IMREAD_GRAYSCALE, 802,531)

# LBP 기반 얼굴검지
lbp_cascade = cv2.CascadeClassifier('../models/lbpcascade_frontalface_improved.xml')
lbp_faces = lbp_cascade.detectMultiScale(img2, 1.5, 1)

res2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
for (x, y, w, h) in lbp_faces:
    cv2.rectangle(res2, (x, y), (x + w, y + h), (0, 0, 255), 2)

displays = [("input2", img2),
            ("res2", res2)]
for (name, out) in displays:
    cv2.imshow(name, out)

# 키보드 입력을 기다린 후 모든 영상창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()
