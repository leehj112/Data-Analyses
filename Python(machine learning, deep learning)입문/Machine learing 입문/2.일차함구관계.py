# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:18:31 2024

@author: leehj
"""

# ## 일차함수 관계식 찾기

# x 변수, y 변수 데이터 만들기 (리스트 객체)

# In[1]:


x = [-3,  31,  -11,  4,  0,  22, -2, -5, -25, -14]
y = [ -2,   32,   -10,   5,  1,   23,  -1,  -4, -24,  -13]
print(x)
print(y)


# 그래프 그리기 (matplotlib)

# In[2]:


import matplotlib.pyplot as plt
plt.plot(x, y)
plt.show()


# - 판다스 데이터프레임 만들기

# In[3]:


import pandas as pd
df = pd.DataFrame({'X':x, 'Y':y})
df.shape


# In[4]:


df.head()


# In[5]:


df.tail()


# 머신러닝 - 사이킷런 *LinearRegression* 모형

# In[6]:


train_features = ['X']
target_cols = ['Y']

X_train = df.loc[:, train_features]
y_train = df.loc[:, target_cols]

X_train1 = df.loc[:, 'X'] # Series
y_train1 = df.loc[:, 'Y'] # Series

X_train2 = df.loc[:, ['X']] # DataFrame
y_train2 = df.loc[:, ['Y']] # DataFrame


print(X_train.shape, y_train.shape)


# In[7]:

# 
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

#%%

# 해당 모듈위치 확인
import inspect
print(inspect.getfile(LinearRegression))
# C:\anaconda3\envs\YSIT2023\Lib\site-packages\sklearn\linear_model\_base.py

#%%

#%%
# ValueError: Expected 2D array, got 1D array instead:
# lr.fit(X_train1, y_train1)

#%%

# 훈련데이터는 2차원 형태의 판다스 데이터프레임이어야 한다.
lr.fit(X_train2, y_train1)

#%%

# 훈련(트래인)
# X_train: 훈련 데이터
# y_train: 정답 데이터
lr.fit(X_train, y_train)

# In[8]:


lr.coef_, lr.intercept_


# In[9]:


print ("기울기: ", lr.coef_[0][0])
print ("y절편: ", lr.intercept_[0])


# In[10]:

# 예측
import numpy as np

# 1행 1열 형태의 2차원 배열
X_new = np.array(11).reshape(1, 1)

#%%

# 예측1
X_pred = lr.predict(X_new)


# In[11]:


# 예측2
# 11부터 16-1까지 1씩 증가한 값을 가지고 배열 생성
# -1 : 크기에 맞게 행을 생성, 자동으로 5행 생성
X_test = np.arange(11, 16, 1).reshape(-1, 1)
X_test


# In[12]:


y_pred = lr.predict(X_test)
y_pred