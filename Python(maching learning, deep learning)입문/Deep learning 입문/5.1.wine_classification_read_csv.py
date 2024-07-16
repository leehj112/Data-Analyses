# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:25:50 2024

@author: leehj
"""

 딥러닝을 활용한 분류 예측: 와인 품질 등급 판별
# # 분석환경 준비

# In[ ]:


# 필수 라이브러리
import pandas as pd
import numpy as np
import random
import tensorflow as tf

# 랜덤 시드 고정
SEED=12
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)  
print("시드 고정: ", SEED)


# In[ ]:


# 구글 드라이브 마운트
# from google.colab import drive
# drive.mount('/gdrive')


# # 데이터 전처리

# In[ ]:


# 데이콘 사이트에서 다운로드한 CSV파일을 읽어오기
# drive_path = "/gdrive/My Drive/"
drive_path = "./"

train = pd.read_csv(drive_path + "wine_train.csv")
test = pd.read_csv(drive_path + "wine_test.csv")
submission = pd.read_csv(drive_path + "wine_sample_submission.csv")

print(train.shape, test.shape, submission.shape) # (5497, 14) (1000, 13) (1000, 2)

#%%

# 칼럼(index) 삭제
train.drop("index", axis=1, inplace=True)
test.drop("index", axis=1, inplace=True)
submission.drop("index", axis=1, inplace=True)


# In[ ]:


train.head(2)


# In[ ]:


submission.head()


# In[ ]:

# while, red
train['type'].value_counts()

#%%
"""
type
white    4159
red      1338
Name: count, dtype: int64
"""

# In[ ]:

# type이 white이면 1, red이면 0
train['type'] = np.where(train['type']=='white', 1, 0).astype(int)
test['type'] = np.where(test['type']=='white', 1, 0).astype(int)
train['type'].value_counts()

#%%
"""
type
1    4159
0    1338
Name: count, dtype: int64
"""

# In[ ]:

# 타겟 : 정답
# quality : 3~9까지
train['quality'].value_counts()

#%%

quality_counts = train['quality'].value_counts()
print(quality_counts.sort_index()) # 시리즈의 인덱스 정렬

#%%

"""
quality
3      26
4     186
5    1788
6    2416
7     924
8     152
9       5
Name: count, dtype: int64
"""

# In[ ]:

# 원핫 인코딩: 0,1
# 와인 3~9까지 7등급
# 0~6까지로 7등급으로 변환
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(train.loc[:, 'quality'] - 3)
y_train

print('y_train.shape:', y_train.shape) # (5497, 7)

# In[ ]:

# 피처 선택
# 컬럼선택: 맨처음('quality')를 제외하고 남은 컬럼 선택
X_train = train.loc[:, 'fixed acidity':]
X_test = test.loc[:, 'fixed acidity':]

# 정규화
# 피처 스케일링
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)

# 훈련(학습) 데이터
X_train_scaled = scaler.fit_transform(X_train)

#%%

scaler = MinMaxScaler()
scaler.fit(X_test)

# 테스트(검증) 데이터
X_test_scaled = scaler.fit_transform(X_test)

print(X_train_scaled.shape, y_train.shape)
print(X_test_scaled.shape)


#%%
###############################################################################
# # 신경망 학습
###############################################################################

# In[ ]:

print("X_train_scaled.shape: ", X_train_scaled.shape)  # (5497, 12)  

#%%

# 심층 신경망 모델
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 활성함수(softmax) : 0~1사의 값으로 다중분류에 사용, 전체를 더하면 1.0(100%)
def build_model(train_data, train_target):
    model = Sequential()
    model.add(Dense(128, activation='tanh', input_dim=train_data.shape[1])) # 입력(12)
    model.add(Dropout(0.2)) # 은닉층 레이어 간의 연결은 제거(20%), 과적합 방지
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(train_target.shape[1], activation='softmax')) # 다중분류(확률) : 출력(7)

    # RMSprop(Root Mean Square Propagation)
    # 기울기 갱신 지수 가중 이동 평균 학습률 조정
    # 현재 기울기를 해당 이동 평균의 제곱근 나누어...
    model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['acc', 'mae'])

    return model

model = build_model(X_train_scaled, y_train)

#%%

model.summary()

#%%

"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 128)               1664       # 입력(12) * 128 + 128
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 64)                8256      
                                                                 
 dropout_1 (Dropout)         (None, 64)                0         
                                                                 
 dense_2 (Dense)             (None, 32)                2080      
                                                                 
 dense_3 (Dense)             (None, 7)                 231       
                                                                 
=================================================================
Total params: 12231 (47.78 KB)
Trainable params: 12231 (47.78 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
"""


#%%
###############################################################################
# # 콜백 함수
###############################################################################

# In[ ]:

# 콜백함수: Early Stopping 기법
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

X_tr, X_val, y_tr, y_val = train_test_split(X_train_scaled, y_train, test_size=0.15, 
                                            shuffle=True, random_state=SEED)

# 검증손실
# patience : 주어진 에포크 동안 연속하여 검증 데이터에 대한 손신함수가 줄어들지 않으면 학습을 종료한다.
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# 배치사이즈: 64
# 에포크(훈련횟수): 200
history = model.fit(X_tr, y_tr, batch_size=64, epochs=200,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping],                    
                    verbose=2)


# In[ ]:


# 평가: 테스트 데이터
model.evaluate(X_val, y_val)
# loss='categorical_crossentropy'
# metrics=['acc', 'mae'])
# [1.0263588428497314, 0.5600000023841858, 0.16253086924552917]

#%%

y_val_pred = model.predict(X_val)

# In[ ]:


# test 데이터에 대한 예측값 정리
y_pred_proba = model.predict(X_test_scaled)
y_pred_proba[:5]



#%%

# 학습 데이터에 대한 예측값
y_train_pred_proba = model.predict(X_tr)

# np.argmax() : 배열에서 가장 큰 값이 있는 원소의 인덱스
# asix: -1, 
#   - 배열의 마지막 축
#   - 2차원이면 각 행에 대해서
#   - 1차원이면 열에 대해서
# 3을 더한이유: 정답에서 -3을 했기 때문에 다시 3을 더해줌
y_train_pred_label = np.argmax(y_train_pred_proba, axis=-1) + 3

loss, acc, mae = model.evaluate(X_tr, y_tr)
print("loss:", loss)
print("accuracy:", acc)
print("mae:", mae)


# In[ ]:

# test 데이터에 대한 예측값의 정답
y_pred_label = np.argmax(y_pred_proba, axis=-1) + 3
y_pred_label[:5]



# In[ ]:


# 제출양식에 맞게 정리
submission['quality'] = y_pred_label.astype(int)
submission.head()


# In[ ]:


# 제출파일 저장    
submission.to_csv(drive_path + "wine_dnn_001.csv", index=False)   


# In[ ]:

import matplotlib.pyplot as plt

# 훈련손실, 검증손실 비교 그래프
def plot_loss_curve(hist, total_epoch=10, start=1):
    histlen = len(hist.history['loss'])
    print('총 에포크 횟수:', histlen)
    
    total_epoch = total_epoch if total_epoch < histlen else histlen
    
    plt.figure(figsize=(5, 5))
    plt.plot(range(start, total_epoch + 1), 
             hist.history['loss'][start-1:total_epoch], # 훈련손실
             label='Train')
    plt.plot(range(start, total_epoch + 1), 
             hist.history['val_loss'][start-1:total_epoch], # 검증손실
             label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('mae')
    plt.legend()
    plt.show()

#%%

#
plot_loss_curve(history, total_epoch=200, start=1)


#%%

# 손실과 정확도

def plot_acc_curve(hist):
    fig, ax1 = plt.subplots(figsize=(6,4))
    ax1.plot(hist.history['loss'], 'r-', label='loss') # 손실
    ax1.set_ylabel('loss')
    ax1.set_xlabel("Epochs")
    ax1.legend()
    
    ax2 = ax1.twinx()
    ax2.plot(hist.history['val_acc'], 'b-', label='acc') # 정확도
    ax2.set_ylabel('acc')
    ax2.legend()
    plt.show()
    
#%%    

plot_acc_curve(history)
    