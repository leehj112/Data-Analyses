# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:25:03 2024

@author: leehj
"""

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

print(train.shape, test.shape, submission.shape)


# In[ ]:


train.head(2)


# In[ ]:


submission.head()


# In[ ]:


train['type'].value_counts()


# In[ ]:


train['type'] = np.where(train['type']=='white', 1, 0).astype(int)
test['type'] = np.where(test['type']=='white', 1, 0).astype(int)
train['type'].value_counts()


# In[ ]:


train['quality'].value_counts()


# In[ ]:


from tensorflow.keras.utils import to_categorical

y_train = to_categorical(train.loc[:, 'quality'] - 3)
y_train


# In[ ]:


# 피처 선택
X_train = train.loc[:, 'fixed acidity':]
X_test = test.loc[:, 'fixed acidity':]

# 피처 스케일링
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

print(X_train_scaled.shape, y_train.shape)
print(X_test_scaled.shape)


# # 신경망 학습

# In[ ]:


# 심층 신경망 모델
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_model(train_data, train_target):
    model = Sequential()
    model.add(Dense(128, activation='tanh', input_dim=train_data.shape[1]))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(train_target.shape[1], activation='softmax'))

    model.compile(optimizer='RMSprop', loss='categorical_crossentropy', 
                metrics=['acc', 'mae'])

    return model

model = build_model(X_train_scaled, y_train)
model.summary()


# # 콜백 함수

# In[ ]:


# Early Stopping 기법
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

X_tr, X_val, y_tr, y_val = train_test_split(X_train_scaled, y_train, test_size=0.15, 
                                            shuffle=True, random_state=SEED)

early_stopping = EarlyStopping(monitor='val_loss',  patience=10)
history = model.fit(X_tr, y_tr, batch_size=64, epochs=200,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping],                    
                    verbose=2)


# In[ ]:


model.evaluate(X_val, y_val)


# In[ ]:


# test 데이터에 대한 예측값 정리
y_pred_proba = model.predict(X_test_scaled)
y_pred_proba[:5]


# In[ ]:


y_pred_label = np.argmax(y_pred_proba, axis=-1) + 3
y_pred_label[:5]


# In[ ]:


# 제출양식에 맞게 정리
submission['quality'] = y_pred_label.astype(int)
submission.head()


# In[ ]:


# 제출파일 저장    
submission.to_csv(drive_path + "wine/wine_dnn_001.csv", index=False)   


# In[ ]: