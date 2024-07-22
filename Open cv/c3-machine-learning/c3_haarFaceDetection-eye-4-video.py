# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:29:21 2024

@author: leehj
"""


# https://github.com/kipr/opencv

# 얼굴 검출 예제 (c3_haarFaceDetection-eye-4-video.py)

# 관련 라이브러리 선언
import cv2
from imgRead import imgRead
from createFolder import createFolder
import time

CAMERA_ID = 0

# cam = cv2.VideoCapture("../images/woman.mp4")
cam = cv2.VideoCapture("../images/boyngirl.mp4")
# cam = cv2.VideoCapture(CAMERA_ID)
if cam.isOpened() == False:
    print('Can not open the(%d)' % (CAMERA_ID) )
    exit()

cv2.namedWindow("CAM_Window") 

#%%

# 얼굴,눈 검지를 위한 모델 설정
face_cascade = cv2.CascadeClassifier('../models/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../models/haarcascade_eye.xml')

#%%

# 시간출력
def dispTime(frame):
    now = time.localtime()
    str = "%d. %d. %d. %d:%d:%d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

    cv2.putText(frame, str, (0, 50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0,0,255))

#%%

# 얼굴과 눈 검지하여 영역 출력
def detectFaceFrame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 얼굴검지
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
 
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2) # 원본 이미지에 검지한 얼굴 사각형 그림
        face_gray = gray[y:y+h, x:x+w]   # 얼굴 영역에서 눈을 검지하기 위한 데이터
        
        # 넘파이 배열 : 원본에서 슬라이싱을 해서 슬라이싱 된 변수의 값을 변화시키면 원본도 변경 됨
        face_frame = frame[y:y+h, x:x+w] # 얼굴 영영에서 눈을 그리기 위한 영역

        # 눈 검지
        eyes = eye_cascade.detectMultiScale(face_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_frame, (ex, ey), (ex+ew, ey + eh), (0,255,255), 2)    
            
#%%

# 메인 루프
while True:
    ret, frame = cam.read()
    if ret != True:
        break
    
    dispTime(frame)        # 이미지 프레임에 시간 출력 
    detectFaceFrame(frame) # 이미지 프레임에 얼굴 영역과 눈 영역을 출력
    
    cv2.imshow('CAM_Window', frame)

    key = cv2.waitKey(30) & 0xFF
    if(key == 27): # Escape Keyboard
        break    
            

#%%
cam.release() # 카메라 해제
cv2.destroyWindow("CAM_Window") # 창 닫기