# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:01:21 2024

@author: leehj
"""

# 블러링 예제 (c2_blur.py)
# 흐릿하게 처리, 영상을 부드럽게 처리

# 관련 라이브러리 선언
import numpy as np
import cv2
from imgRead import imgRead
from createFolder import createFolder

# 영상 읽기
img1 = imgRead("../images/img8.jpg", cv2.IMREAD_GRAYSCALE, 320, 240)

# 필터 정의 및 블러링 수행
# 필터의 크기 : W * H
ksize1 = 3; ksize2 = 5; ksize3 = 7; ksize4 = 9

#kernel = np.full(shape=[ksize4,ksize4], fill_value=1, dtype=np.float32) / (ksize4*ksize4)
kernel = np.full(shape=[ksize4,ksize4], fill_value=1, dtype=np.float32) / (ksize4*ksize4*4)
# kernel = np.full(shape=[ksize4,ksize4], fill_value=1)

#%%

res1 = cv2.blur(img1, (ksize1,ksize1))
res2 = cv2.blur(img1, (ksize2,ksize2))

res3 = cv2.boxFilter(img1, -1, (ksize3,ksize3))

# normalize: False이면 필터가 적용된 값이 255에 가까워 지거나 큰값을 갖게 됨
# res3 = cv2.boxFilter(img1, -1, (ksize3,ksize3), normalize=False)

res4 = cv2.filter2D(img1, -1, kernel)
#res5 = cv2.boxFilter(img1, -1, (1,21))
res5 = cv2.boxFilter(img1, -1, (21,1))

# 결과 영상 출력
displays = [("input1", img1),
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
save_dir = '../code_res_imgs/c2_blur'
createFolder(save_dir)
for (name, out) in displays:
    cv2.imwrite(save_dir+"/"+name+".jpg", out)