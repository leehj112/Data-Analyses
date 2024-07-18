# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:29:58 2024

@author: leehj
"""

# 비지도 학습: 정답(타깃)을 모르는 상태에서 종류 분류 
# 군집 알고리즘
# 데이터셋: 과일 사진 데이터 준비하기
# 과일의 종류 : 사과(apple), 파인애플(pineapple), 바나나(banana)
# 과일사진 : 300장 * 가로(100) * 세로(100)

# 데이터셋
# https://www.kaggle.com/datasets/moltean/fruits?resource=download
# 웹브라우저에서 https://bit.ly/fruits_300_data
# get_ipython().system('wget https://bit.ly/fruits_300_data -O fruits_300.npy')

#%%

import numpy as np
import matplotlib.pyplot as plt


#%%


fruits = np.load('./fruits_300.npy')


# In[21]:

# 과일사진 : 300장 * 가로(100) * 세로(100)
print(fruits.shape) # (300, 100, 100)


# In[22]:


# 첫 번째 이미지의 첫 번째 행의 데이터
# 값(흑백) : 0 ~ 255, 검정(0), 흰색(255)
print(fruits[0, 0, :])

#%%

for x in range(100):
    print(fruits[0, x, :])
    


# In[23]:

# 저장된 이미지는 원래 이미지를 반전시켜서 변환
# 학습을 위해서 : 배경이 검은색(0)에 가까운 색으로 변환
# 첫 번째 이미지 출력
plt.imshow(fruits[0])
plt.imshow(fruits[0], cmap='gray')
plt.show()


# In[24]:

# 사람이 인식하는 형태 : 배경이 흰색    
# 반전시켜서 그레이 형태로 출력
plt.imshow(fruits[0], cmap='gray_r')
plt.show()


# In[25]:


# 사과, 파인애플, 바나나
fig, axs = plt.subplots(3, 2)

# 사과
axs[0,0].imshow(fruits[0], cmap='gray_r')
axs[0,1].imshow(fruits[99], cmap='gray_r')

# 파인애츨
axs[1,0].imshow(fruits[100], cmap='gray_r')
axs[1,1].imshow(fruits[199], cmap='gray_r')

# 바나나
axs[2,0].imshow(fruits[200], cmap='gray_r')
axs[2,1].imshow(fruits[299], cmap='gray_r')
plt.show()


#%%
# ## 픽셀 값 분석하기

# In[26]:

# 각 이미지 종류별로 2차원(100 * 10000) 배열로 변환
# 애플, 파인애플, 바나나
# 과일의 갯수는 100개 씩 분포
apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)


# In[27]:

print(apple.shape)     # (100, 10000)
print(pineapple.shape) # (100, 10000)
print(banana.shape)    # (100, 10000)


# In[28]:

# 각 이미지별 평균 : 10000개 픽셀의 평균
# 행단위로 모든 열(10000개)의 평균
# 100행의 평균
print(apple.mean(axis=1))

#%%

apple_mean = apple.mean(axis=1)
pinea_mena = pineapple.mean(axis=1)
banana_mean = banana.mean(axis=1)


# In[29]:

# 각 과일의 평균
# 히스토그램
# x : 값의 구간(계급), 평균
# y : 발생 빈도(도수)    
plt.hist(np.mean(apple, axis=1), alpha=0.8)
plt.hist(np.mean(pineapple, axis=1), alpha=0.8)
plt.hist(np.mean(banana, axis=1), alpha=0.8)
plt.legend(['apple', 'pineapple', 'banana'])
plt.show()

#%%

plt.hist(apple_mean, alpha=0.8)
plt.hist(pinea_mena, alpha=0.8)
plt.hist(banana_mean, alpha=0.8)
plt.legend(['apple', 'pineapple', 'banana'])
plt.show()


#%%

# 픽셀별 평균: 세로축, 각 셀단위    
# 각 과일별 100개의 이미지의 픽셀별 평균    
apple_axis = np.mean(apple, axis=0)    

#%%

fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].bar(range(10000), np.mean(apple, axis=0))
axs[1].bar(range(10000), np.mean(pineapple, axis=0))
axs[2].bar(range(10000), np.mean(banana, axis=0))
plt.show()


# In[31]:

# 벡터(10000) -> 매트릭트(100 * 100)
# 각 과일의 종류별로 픽셀별 평균으로 이미지화
# 100장의 사진의 평균 이미지
apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
banana_mean = np.mean(banana, axis=0).reshape(100, 100)

print(apple_mean.shape)     # (100, 100)
print(pineapple_mean.shape) # (100, 100)
print(banana_mean.shape)    # (100, 100)

#%%

# 픽셀별 평균을 100 * 100의 이미지로 표현
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].imshow(apple_mean, cmap='gray_r')
axs[1].imshow(pineapple_mean, cmap='gray_r')
axs[2].imshow(banana_mean, cmap='gray_r')
plt.show()

#%%

###############################################################################
# ## 평균값과 가까운 사진 고르기
###############################################################################

# 원본사진에서 픽셀별 평균을 가지고 오차가 가장 작은 샘플 선택

# In[32]:

abs_diff = np.abs(fruits - apple_mean) # 절대값
abs_mean_apple = np.mean(abs_diff, axis=(1,2)) # 행과 열에 대한 차(전체-애플의 평균)의 평균
print(abs_mean_apple.shape)

#%%

# 애플 평균값의 오름차순으로 인덱스를 구함
apple_index = np.argsort(abs_mean_apple)[:100]
apple_index_sort = np.sort(apple_index)

#%%
fig, axs = plt.subplots(10, 10, figsize=(10,10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[apple_index[i*10 + j]], cmap='gray_r')
        axs[i, j].axis('off')
plt.show()

#%%

# ## 확인문제
# 바나나
abs_diff = np.abs(fruits - banana_mean)
abs_mean_banana = np.mean(abs_diff, axis=(1,2))

# 바나나 평균값의 오름차순으로 인덱스를 구함
banana_index = np.argsort(abs_mean_banana)[:100]
banana_index_sort = np.sort(banana_index)

fig, axs = plt.subplots(10, 10, figsize=(10,10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[banana_index[i*10 + j]], cmap='gray_r')
        axs[i, j].axis('off')
plt.show()


#%%

# ## 확인문제
# 파인애플

abs_diff = np.abs(fruits - pineapple_mean)
abs_mean_pineapple = np.mean(abs_diff, axis=(1,2))

# 파인애플 평균값의 오름차순으로 인덱스를 구함
pineapple_index = np.argsort(abs_mean_pineapple)[:100]
pindapple_index_sort = np.sort(pineapple_index)


fig, axs = plt.subplots(10, 10, figsize=(10,10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[pineapple_index[i*10 + j]], cmap='gray_r')
        axs[i, j].axis('off')
plt.show()