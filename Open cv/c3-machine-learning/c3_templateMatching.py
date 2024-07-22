# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:31:55 2024

@author: leehj
"""

#템플릿 매칭을 통한 객체 검출 예제 (c3_templateMatching.py)

# 관련 라이브러리 선언
import numpy as np
import cv2
from matplotlib import pyplot as plt
from imgRead import imgRead
from createFolder import createFolder

#%%

es = "10 + 30"
er = eval(es) # 파이썬 코드를 실행
print(er) # 40

#%%

print("cv2.TM_SQDIFF:", cv2.TM_SQDIFF)
print("cv2.TM_SQDIFF:", eval("cv2.TM_SQDIFF"))

#%%

# 영상 읽기
img1 = imgRead("../images/img_6_0.png", cv2.IMREAD_GRAYSCALE, 320, 240)

# 탬플릿 매칭 및 결과 출력/저장
template = img1[5:70, 5:70]
w, h = template.shape[::-1] # 65 ,65
methods = ['cv2.TM_CCOEFF',
           'cv2.TM_CCOEFF_NORMED',
           'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED',
           'cv2.TM_SQDIFF',
           'cv2.TM_SQDIFF_NORMED']
save_dir = './code_res_imgs/c3_templateMatching'
createFolder(save_dir)
cv2.imwrite(save_dir + "/" + "template.png", template)

for meth in methods:
    input = img1.copy()
    method = eval(meth)
    print(f"method: {method} -> {meth}")
    res = cv2.matchTemplate(img1, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    print(f"[res] {min_val}, {max_val}, {min_loc}, {max_loc}")

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
        
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(input, top_left, bottom_right, 0, 2)

    plt.figure()
    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(122), plt.imshow(input, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth) 

    # 영상 저장
    plt.savefig(save_dir + "/" + meth + ".png")
    res = np.abs(res) ** 3
    _val, res = cv2.threshold(res, 0.01, 0, cv2.THRESH_TOZERO)
    res8 = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite(save_dir + "/" + meth + "_res.png", res8)
    cv2.imwrite(save_dir + "/" + meth + "_img.png", input)

plt.show()