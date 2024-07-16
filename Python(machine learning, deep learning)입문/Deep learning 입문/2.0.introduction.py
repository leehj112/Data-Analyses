# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:23:29 2024

@author: leehj
"""

# ### 텐서플로 2.0 
# tensorflow : 2.14.0

# In[ ]:


import tensorflow as tf
print(tf.__version__)


# ### x 변수, y 변수 데이터 

# In[ ]:


import pandas as pd
import numpy as np

x = [-3,  31,  -11,  4,  0,  22, -2, -5, -25, -14]
y = [ -2,   32,   -10,   5,  1,   23,  -1,  -4, -24,  -13]

X_train = np.array(x).reshape(-1, 1) # 훈련
y_train = np.array(y)                # 정답  

print(X_train.shape, y_train.shape)


# ### 인공 신경망 모델 

# In[ ]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(units=1, activation='linear', input_dim=1))


# In[ ]:


model.summary()


# In[ ]:


model.compile(optimizer='adam', loss='mse', metrics=['mae'])


# In[ ]:


# model.fit(X_train, y_train, epochs=3000, verbose=0)
model.fit(X_train, y_train, epochs=3000, verbose=1)


# In[ ]:


model.weights


# ### 예측

# In[ ]:


model.predict([[11], [12], [13]])