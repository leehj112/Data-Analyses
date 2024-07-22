# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:27:30 2024

@author: leehj
"""

# 코너 검출 예제 (c3_cornerHarris.py)

# 관련 라이브러리 선언
import numpy as np
import cv2
from imgRead import imgRead
from createFolder import createFolder

# 영상 읽기
img1 = imgRead("../images/img_6_4.png", cv2.IMREAD_GRAYSCALE, 320, 240)

# 코너 검지
# 블럭사이즈:
# 비교하는 이웃 픽셀의 크기를 증가시키면 밝기 변화가 큰 영역이 포함될 가능성이 높다.
blockSize = 5
dst = cv2.cornerHarris(img1, blockSize, 3, 0.06)
dst = cv2.dilate(dst,None)
res1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
res1[dst>0.1*dst.max()] = [0,0,255]

# 결과 영상 출력
displays = [("input1", img1),
            ("res1", res1)]
for (name, out) in displays:
    cv2.imshow(name, out)

# 키보드 입력을 기다린 후 모든 영상창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()

# 영상 저장
save_dir = "../code_res_imgs/c3_cornerHarris"
createFolder(save_dir)
for (name, out) in displays:
    cv2.imwrite(save_dir + "/" + name + ".jpg", out)