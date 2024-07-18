# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:33:22 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# [주성분 분석]
#   - 비지도학습에 속한다.
#   - 종속변수가 존재하지 않는다.
#   - 예측이나 분류하지 않는다.
#   - 데이터의 차원을 축소
#   - 독립변수의 갯수를 줄임
#   - 기존의 특정을 보존
#   - 기존 변수들의 정보를 모두 반영하여 새로운 변수를 만드는 방식으로 차원 축소
#   - 다차원을 2차원으로 축소
#   - 변수 간의 상관관계 문제를 해결
#   - 단점
#     . 기존 변수를 새로운 변수로 변환했을 때 해석의 어려움
#     . 차원이 축소되면 정보 손실 발생
#   - 활용
#     . 다차원 변수를 2차원으로 표현
#     . 변수가 너무 많은 경우(차원츨 축소하여 모댈 학습 시간을 절약)
#     . 오버피팅(과대적합) 방지
#   - 특징
#     . 데이터에서 가장 분산이 큰 방향을 찾는다.
#   - 설명된 분산
#     . 주성분 분석에서 주성분이 얼마나 원본 데이터의 분산을 잘 나타내는가?

# PCA(Principal Component Analysis) 클래스
# 차원축소 알고리즘 : 특성의 축소
# 이미지 샘플 예제: 
#  1. 하나의 이미지는 100 * 100으로 구성된 10000의 특성을 가지고 있다.
#  2. 주성분만 추출하여 특성을 줄이면 모델의 성능 향상    

#%%

import numpy as np

fruits = np.load('./fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)


#%%
from sklearn.decomposition import PCA

help(PCA)
#%%
# 주성분 50개 지정 : n_components=50
pca = PCA(n_components=50)

# 주요성분 분석
pca.fit(fruits_2d)


#%%

print(pca.components_.shape) # (50, 10000)


#%%

import matplotlib.pyplot as plt

def draw_fruits(arr, ratio=1):
    n = len(arr)    # n은 샘플 개수입니다
    # 한 줄에 10개씩 이미지를 그립니다. 샘플 개수를 10으로 나누어 전체 행 개수를 계산합니다.
    rows = int(np.ceil(n/10))
    # 행이 1개 이면 열 개수는 샘플 개수입니다. 그렇지 않으면 10개입니다.
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols,
                            figsize=(cols*ratio, rows*ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:    # n 개까지만 그립니다.
                axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()


#%%

# 
# 주성분은 원본 데이터에서 가장 분산이 큰 방향을 순서대로 나타낸 것이다.
draw_fruits(pca.components_.reshape(-1, 100, 100))


#%%

print(fruits_2d.shape) # (300, 10000)

#%%

# 원본 데이터를 주성분에 투영하여 특성의 갯수(10000)를 50개로 축소
# 차원축소 : 데이터의 차원을 50으로 줄임
fruits_pca = pca.transform(fruits_2d)

#%%

# 1/200로 축소 : (300,10000) -> (300, 50)
# 50개의 특성을 가진 데이터
print(fruits_pca.shape) # (300, 50)


#%%

###############################################################################
# 원본 데이터 재구성
###############################################################################

#%%

# 원본 데이터로 재구성
fruits_inverse = pca.inverse_transform(fruits_pca)
print(fruits_inverse.shape) # (300, 10000)

#%%

# 100 * 100 크기로 재구성
fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)


#%%

for start in [0, 100, 200]:
    draw_fruits(fruits_reconstruct[start:start+100])
    print("\n")


#%%
# 설명된 분산(Explained Variance)
# 주성분이 원본 데이터의 분산을 얼마나 잘 나타내는지 기록한 값

#%%

# 주성분의 설명된 분산 비율
print(np.sum(pca.explained_variance_ratio_)) # 0.9214092598363355


# In[14]:

# x: 주성분의 갯수
# y: 분산
# ※ 주성분 10개가 대부분의 분산을 표현
plt.plot(pca.explained_variance_ratio_)


#%%

###############################################################################
# 다른 알고리즘과 함께 사용하기
# 과일 사진 원본 데이터와 PCA로 축소한 데이터를 지도학습에 적용해서
# 어떤 차이가 있는지 알아보자.

# In[15]:


# 로지스틱 회귀 모델로 지도학습
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()


# In[16]:

# 정답 데이터를 만듦: 총 300개 = 0(100개), 1(100개), 2(100개)
target = np.array([0] * 100 + [1] * 100 + [2] * 100)


# In[17]:


from sklearn.model_selection import cross_validate

# 정확도 99%, 훈련시간 : 0.221초
scores = cross_validate(lr, fruits_2d, target)
print(np.mean(scores['test_score'])) # 0.9966666666666667
print(np.mean(scores['fit_time']))   # 0.22182302474975585 초


# In[18]:

# 훈련시간이 단축
# 정확도 99%, 훈련시간 : 0.00312초
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score'])) # 0.9966666666666667
print(np.mean(scores['fit_time']))   # 0.003124427795410156


# In[19]:

# 설명된 분산의 50%에 해당하는 주성분을 찾아라.
pca2 = PCA(n_components=0.5)
pca2.fit(fruits_2d)


# In[20]:

# 주성분 2개
print(pca2.n_components_) # 2개


# In[21]:

# 차원축소
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape) # (300, 2)


# In[22]:

# 교차검증
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score'])) # 0.9933333333333334
print(np.mean(scores['fit_time']))   # 0.018746137619018555


# In[23]:

###############################################################################
# k-평균 알고리즘
###############################################################################
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_pca)


# In[24]:


print(np.unique(km.labels_, return_counts=True)) 
# (array([0, 1, 2]), array([110,  99,  91], dtype=int64))


# In[25]:


for label in range(0, 3):
    draw_fruits(fruits[km.labels_ == label])
    print("\n")


# In[26]:

# 사과와 파인애플은 클러스터의 경계가 근접해 있다.
for label in range(0, 3):
    data = fruits_pca[km.labels_ == label]
    plt.scatter(data[:,0], data[:,1])
plt.legend(['apple', 'banana', 'pineapple'])
plt.show()
