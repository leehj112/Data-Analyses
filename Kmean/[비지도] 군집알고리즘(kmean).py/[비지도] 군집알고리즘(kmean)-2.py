# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:33:02 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# # k-평균
# ## KMeans 클래스

import numpy as np

fruits = np.load('./fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)


# In[3]:


from sklearn.cluster import KMeans

# 3개의 종류로 분류: 애플, 파인애플, 바나나 
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)


# In[4]:


print(km.labels_)


# In[5]:


print(np.unique(km.labels_, return_counts=True))


# In[6]:


import matplotlib.pyplot as plt

def draw_fruits(arr, ratio=1):
    n = len(arr)    # n은 샘플 개수입니다
    
    # 한 줄에 10개씩 이미지를 그립니다. 
    # 샘플 개수를 10으로 나누어 전체 행 개수를 계산합니다.
    rows = int(np.ceil(n/10))
    
    # 행이 1개 이면 열 개수는 샘플 개수입니다. 
    # 그렇지 않으면 10개입니다.
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols,
                            figsize=(cols*ratio, rows*ratio), squeeze=False)
    
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:    # n 개까지만 그립니다.
                axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()


# In[7]:


draw_fruits(fruits[km.labels_==0]) # 사과


# In[8]:


draw_fruits(fruits[km.labels_==1]) # 바나나


# In[9]:


draw_fruits(fruits[km.labels_==2]) # 파인애플

#%%

# ## 클러스터 중심
# 애플, 바나나, 파일애플
draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3)


# In[11]:

# 100번째 샘플의 클러스터 중심까지의 거리
print(km.transform(fruits_2d[100:101]))

# 거리가 가장 가가운 것은 2번째
# [[5267.70439881 8837.37750892 3393.8136117 ]]

# In[12]:


print(km.predict(fruits_2d[100:101])) # [2]


# In[13]:


draw_fruits(fruits[100:101]) # 파인애플


# In[14]:

# k-평균 알고리즘이 최적의 클러스터를 찾아 반복한 횟수
print(km.n_iter_) # 4


#%%
# ## 최적의 k 찾기
# 사전에 클러스터의 갯수를 지정할 수 없는 경우
# 이너셔(inertia) : 클러스터의 샘플이 얼마나 가깝게 있는지를 나타내는 값
# 엘보우(elbow) : 클러스터의 갯수를 늘려가면서 이너셔의 변화를 관찰하여
#                 최적의 클러스터 갯수를 찾는 방법
# 클러스터 갯수가 늘어나면 이너셔의 갯수는 줄어든다.

# In[15]:



inertia = []
for k in range(2, 7): # 클러스터의 갯수
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    km.fit(fruits_2d)
    inertia.append(km.inertia_)

plt.plot(range(2, 7), inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()

# 결과는 클러스터 3의 지점에서 꺾임
