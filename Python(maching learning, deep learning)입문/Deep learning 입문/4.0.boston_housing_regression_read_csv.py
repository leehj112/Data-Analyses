# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:24:33 2024

@author: leehj
"""

# 딥러닝
# 보스턴 주택 가격 예측 예제

# # 라이브러리 환경

# In[1]:


import pandas as pd
import numpy as np
import random
import tensorflow as tf
print(tf.__version__) # 2.14.0


# In[2]:


# 랜덤 시드 고정
SEED=12
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)  
print("시드 고정: ", SEED)


# # 데이터 전처리

# In[3]:


# skleran 데이터셋에서 보스턴 주택 데이터셋 로딩
from sklearn import datasets
# housing = datasets.load_boston()

# CSV 파일에서 데이터셋 로딩
X_data = pd.read_csv("./boston_housing.csv")
y_data = pd.read_csv("./boston_housing_target.csv")

#%%
# X_data = housing.data
# y_data = housing.target
print(X_data.shape, y_data.shape) # (506, 13) (506, 1)


# In[17]:


# housing.feature_names


# In[ ]:
    
# StandardScaler 
# 클래스는 데이터의 특성(feature)들을 
# 평균이 0이고 표준편차가 1인 정규분포로 변환하는 스케일링 작업을 수행
# 각 특성의 분포가 평균 0, 표준편차 1인 정규분포에 가까워지게 됩니다.

# 0~1 사의 값으로 정규화
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_data_scaled = scaler.fit_transform(X_data)    

# 학습 - 테스트 데이터셋 분할
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data_scaled, y_data, 
                                                    test_size=0.2, 
                                                    shuffle=True, 
                                                    random_state=SEED)
print(X_train.shape, y_train.shape) # (404, 13) (404, 1)
print(X_test.shape, y_test.shape)   # (102, 13) (102, 1)


# # 신경망 학습

# In[ ]:


# 심층 신경망
from tensorflow.keras import Sequential    # 케라스의 신경망 모델 클래스
from tensorflow.keras.layers import Dense  # 밀집층을 만드는 클래스

# Dense(뉴런수, 활성함수, ...)

def build_model(num_input=1):
    model = Sequential() # 모델
    
    # 입력층
    model.add(Dense(128, activation='relu', input_dim=num_input)) 
    
    # 은닉층
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    
    # 출력층
    model.add(Dense(1, activation='linear')) # 출력계층

    # 옵티마이저, 손실함수, 측정지표 등을 지정
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model

model = build_model(num_input=13)

#%%
model.summary()

#%%
"""
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_10 (Dense)            (None, 128)               1792      # 입력파라미터(13) * 뉴런(128) + 뉴런(128)
                                                                 
 dense_11 (Dense)            (None, 64)                8256      # 이전뉴런(128) * 뉴런(64) + 뉴런(64)
                                                                 
 dense_12 (Dense)            (None, 32)                2080      # 이전뉴런(64) * 뉴런(32) + 뉴런(32)
                                                                 
 dense_13 (Dense)            (None, 16)                528       # 이전뉴런(32) * 뉴런(16) + 뉴런(16)
                                                                 
 dense_14 (Dense)            (None, 1)                 17        # 이전뉴런(16) * 뉴런(1) + 뉴런(1)
                                                                 
=================================================================
Total params: 12673 (49.50 KB)
Trainable params: 12673 (49.50 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
"""


# In[ ]:

# 모델 훈련
# 1 에포크 : 12번 <- 훈련데이터(404) / 배치(32)
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2)


# In[ ]:


# 평가
model.evaluate(X_test, y_test) 
# mse, mae
# [12.536484718322754, 2.62740159034729]


#%%
###############################################################################
# # 교차 검증
###############################################################################

model = build_model(num_input=13)
history = model.fit(X_train, y_train, batch_size=32, epochs=200, validation_split=0.25, verbose=2)
# history = model.fit(X_train, y_train, batch_size=32, epochs=200, verbose=2)


#%%

import matplotlib.pyplot as plt

def plot_loss_curve(total_epoch=10, start=1):
    plt.figure(figsize=(5, 5))
    plt.plot(range(start, total_epoch + 1), 
             history.history['loss'][start-1:total_epoch], # 훈련손실
             label='Train')
    plt.plot(range(start, total_epoch + 1), 
             history.history['val_loss'][start-1:total_epoch], # 검증손실
             label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('mse')
    plt.legend()
    plt.show()

plot_loss_curve(total_epoch=200, start=1)


# In[ ]:


plot_loss_curve(total_epoch=200, start=20)