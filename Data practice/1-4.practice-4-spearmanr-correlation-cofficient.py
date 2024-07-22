# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:26:19 2024

@author: leehj
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 09:16:12 2024

@author: Solero
"""

#%%
# 스피어만 상관계수
# 스피어만 상관계수는 두 변수 간의 비선형 관계를 측정하는 방법 중 하나이다. 
# 이는 각 데이터 값을 순위로 변환한 후, 순위 간의 피어슨 상관계수를 계산하여 구한다.

import numpy as np
from scipy.stats import spearmanr

# 예시 데이터 생성
# x = np.array([10, 20, 30, 40, 50])
# y = np.array([15, 25, 35, 45, 55])

x = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110])
y = np.array([5,  10, 15, 20, 40, 80, 160, 320, 640, 1280, 2520])


# 스피어만 상관계수 계산
corr, p_value = spearmanr(x, y)

print(f"스피어만 상관계수: {corr}")
print(f"p-value: {p_value}")

# 결과 해석
if p_value < 0.05:
    print("두 변수 간의 상관 관계가 유의미합니다.")
else:
    print("두 변수 간의 상관 관계가 유의미하지 않습니다.")

#%%

# 데이터 시각화

import matplotlib.pyplot as plt

plt.scatter(x, y)
plt.title('Scatter plot of x and y')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#%%
# THE END