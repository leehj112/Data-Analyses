# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:12:41 2024

@author: leehj
"""

# MNIST 추론 예제 (c4_mnistInference.py)

# 관련 라이브러리 선언
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from createFolder import createFolder

# 학습된 모델 불러오기
save_dir = './code_res_imgs/c4_minist'
createFolder(save_dir)
model = load_model(save_dir + '/mnist_model.h5')

# 마우스 필기체 입력 받기
onDown = False
xset, yset = None, None
def onmouse(event, x, y, flags, params):
    global onDown, img1, xset, yset
    if event == cv2.EVENT_LBUTTONDOWN:
        onDown = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if onDown == True:
            cv2.line(img1, (xset,yset), (x,y), (0,0,0), 5)
    elif event == cv2.EVENT_LBUTTONUP:
        onDown = False
    xset, yset = x,y

# 입력 필기체 인식 및 결과 출력
cv2.namedWindow("Input")
cv2.setMouseCallback("Input", onmouse)
width, height = 280, 280
img1 = np.ones((280,280,3), np.uint8) * 255

res_index = 1
while True:
    cv2.imshow("Input", img1)
    key = cv2.waitKey(1)
    if key == ord('r'):
        img1 = np.ones((280,280,3), np.uint8) * 255
    if key == ord('i'):
        x_resize = cv2.resize(img1, dsize=(28,28), interpolation=cv2.INTER_AREA)
        x_gray = cv2.cvtColor(x_resize, cv2.COLOR_BGR2GRAY)
        x_gray = ~x_gray
        x = x_gray.reshape(1, 28, 28, 1)
        y = model.predict_classes(x)
        cv2.putText(img1, str(y), (10,30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (128,128,128), 2)
        cv2.imwrite(save_dir + '/' + str(res_index) + '.jpg', img1)
        res_index += 1
    if key == ord('q'):
        break

# 영상창 닫기
cv2.destroyAllWindows()


