# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:29:05 2024

@author: leehj
"""


# https://github.com/kipr/opencv

# 얼굴 검출 예제 (c3_haarFaceDetection-eye-3.py)

# 관련 라이브러리 선언
import cv2
from imgRead import imgRead
from createFolder import createFolder

# 영상 읽기
img1 = imgRead("../images/img_6_6.jpg", cv2.IMREAD_GRAYSCALE, 640, 426)

#%%
# 얼굴검지
face_cascade = cv2.CascadeClassifier('../models/haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(img1, 1.5, 3)

# 눈 검지
eye_cascade = cv2.CascadeClassifier('../models/haarcascade_eye.xml')


#%%
# 결과 영상 출력
res1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

#%%
for (x, y, w, h) in faces:
    res1 = cv2.rectangle(res1, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = res1[y:y + h, x:x + w]
    roi_color = res1[y:y + h, x:x + w]
    
    # 눈 검지
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey + eh), (0,255,0), 2)

displays = [("input1", img1),
            ("res1", res1)]
for (name, out) in displays:
    cv2.imshow(name, out)

# 키보드 입력을 기다린 후 모든 영상창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()
