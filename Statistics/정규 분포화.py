# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:00:17 2024

@author: leehj
"""

"""
데이터를 변환하여 정규분포에 가깝게 만드는 방법
로그 변환, 제곱근 변환, 역수 변환
"""

#%%

# 로그 변환 (Log Transformation)
# 로그 변환은 데이터의 스케일을 줄이고 분포의 비대칭성을 완화하는 데 유용하다.

import numpy as np
import matplotlib.pyplot as plt

# 예제 데이터
data = np.random.exponential(scale=2, size=1000)

# 로그 변환
log_data = np.log(data + 1)  # 로그 변환 시 0이 포함될 수 있으므로 1을 더해줍니다.

# 히스토그램 시각화
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(data, bins=30)
plt.title('원본 데이터')

plt.subplot(1, 2, 2)
plt.hist(log_data, bins=30)
plt.title('로그 변환 데이터')

plt.show()

#%%

# 제곱근 변환 (Square Root Transformation)
# 제곱근 변환은 데이터의 스케일을 줄이면서 분포의 왜곡을 줄이는 데 도움이 된다.

# 제곱근 변환
sqrt_data = np.sqrt(data)

# 히스토그램 시각화
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(data, bins=30)
plt.title('원본 데이터')

plt.subplot(1, 2, 2)
plt.hist(sqrt_data, bins=30)
plt.title('제곱근 변환 데이터')

plt.show()

#%%

# 역수 변환 (Reciprocal Transformation)
# 역수 변환은 데이터의 크기를 반전시켜 큰 값을 작은 값으로, 작은 값을 큰 값으로 바꾼다.

# 역수 변환
reciprocal_data = 1 / (data + 1)  # 0이 포함될 수 있으므로 1을 더해준다.

# 히스토그램 시각화
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(data, bins=30)
plt.title('원본 데이터')

plt.subplot(1, 2, 2)
plt.hist(reciprocal_data, bins=30)
plt.title('역수 변환 데이터')

plt.show()

#%%

# Box-Cox 변환
# Box-Cox 변환은 로그 변환, 제곱근 변환 등의 일반적인 변환 방법을 일반화한 것으로, 
# 파라미터를 통해 최적의 변환을 찾을 수 있다.

from scipy.stats import boxcox

# Box-Cox 변환 (데이터는 양수여야 합니다)
boxcox_data, best_lambda = boxcox(data + 1)  # 0이 포함될 수 있으므로 1을 더해줍니다.

# 히스토그램 시각화
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(data, bins=30)
plt.title('원본 데이터')

plt.subplot(1, 2, 2)
plt.hist(boxcox_data, bins=30)
plt.title('Box-Cox 변환 데이터')

plt.show()

print('최적의 람다 값:', best_lambda)

#%%

# 이와 같은 변환 방법들은 데이터의 특성에 따라 적절하게 선택하여 사용할 수 있다. 
# 변환 후에는 데이터가 정규분포를 따르는지 확인하기 위해 히스토그램, Q-Q 플롯 등을 활용하거나 
# 정규성 검정을 수행할 수 있습니다.