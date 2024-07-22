# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:12:15 2024

@author: leehj
"""
# 잡음 제거 예제 (c2_noiseCancelling.py)

# 관련 라이브러리 선언
import numpy as np
import cv2
from imgRead import imgRead
from createFolder import createFolder

# 영상 읽기
img1 = imgRead("../images/img10.jpg", cv2.IMREAD_GRAYSCALE, 320, 240)

# 잡음 추가
h, w = img1.shape
sort_pepper_ratio = (np.uint32)((h * w) * 0.1) # 7680
sort_noise_x = np.full(shape=[sort_pepper_ratio], fill_value=0, dtype=np.uint16)
sort_noise_y = np.full(shape=[sort_pepper_ratio], fill_value=0, dtype=np.uint16)
pepper_noise_x = np.full(shape=[sort_pepper_ratio], fill_value=0, dtype=np.uint16)
pepper_noise_y = np.full(shape=[sort_pepper_ratio], fill_value=0, dtype=np.uint16)
gaussian_noise = np.full(shape=[h,w], fill_value=0, dtype=np.uint8)

# 난수발생: 균일분포 난수 생성
# 노이즈 추가
cv2.randu(sort_noise_x, 0, w)
cv2.randu(sort_noise_y, 0, h)
cv2.randu(pepper_noise_x, 0, w)
cv2.randu(pepper_noise_y, 0, h)
cv2.randn(gaussian_noise, 0, 20)

sort_pepper_img = cv2.copyTo(img1, None)
gaussian_noise_img = cv2.add(img1, gaussian_noise)
for i in range(sort_pepper_ratio):
     sort_pepper_img[sort_noise_y[i], sort_noise_x[i]] = 255   # 흰색
     sort_pepper_img[pepper_noise_y[i], pepper_noise_x[i]] = 0 # 검은색

# 잡음 제거
ksize1 = 3; ksize2 = 5
res1 = cv2.medianBlur(sort_pepper_img, ksize1)
res2 = cv2.medianBlur(sort_pepper_img, ksize2)
res3 = cv2.blur(gaussian_noise_img, (ksize1, ksize1))
res4 = cv2.GaussianBlur(gaussian_noise_img, (ksize1, ksize1), 0)
res5 = cv2.bilateralFilter(gaussian_noise_img, -1, 20, ksize1)  # 표준편차를 이용한 양방향 필터

# 결과 영상 출력
displays = [("input1", img1),
            ("noise(sort_prpper) img", sort_pepper_img),
            ("gaussian noise", gaussian_noise),
            ("noise(gaussian) img", gaussian_noise_img),
            ("res1", res1),
            ("res2", res2),
            ("res3", res3),
            ("res4", res4),
            ("res5", res5)]
for (name, out) in displays:
    cv2.imshow(name, out)

# 키보드 입력을 기다린 후 모든 영상창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()

# 영상 저장
save_dir = '../code_res_imgs/c2_noiseCancelling'
createFolder(save_dir)
for (name, out) in displays:
    cv2.imwrite(save_dir + "/" + name + ".jpg", out)