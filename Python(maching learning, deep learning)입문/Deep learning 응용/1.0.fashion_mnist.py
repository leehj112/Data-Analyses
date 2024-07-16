# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:32:46 2024

@author: leehj
"""

#%%

# DNN(Deep Neural Network)

#%%
# # 1. 환경 설정

# In[ ]:


# 라이브러리 설정
import pandas as pd
import numpy as np
import tensorflow as tf
import random

# 랜덤 시드 고정
SEED=12
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)  

# 구글 드라이브 폴더 마운트
from google.colab import drive, files
drive.mount('/gdrive')


# # 2. 데이터셋 준비

# In[ ]:


# 데이콘 사이트에서 다운로드한 CSV파일을 읽어오기
drive_path = "/gdrive/My Drive/"
train = pd.read_csv(drive_path + "mnist_data/train.csv")
test = pd.read_csv(drive_path + "mnist_data/test.csv")
submission = pd.read_csv(drive_path + "mnist_data/sample_submission.csv")

print(train.shape, test.shape, submission.shape)   


# In[ ]:


# train 데이터 보기
train.head()


# In[ ]:


# train 데이터를 28*28 이미지로 변환
train_images = train.loc[:, 'pixel1':].values.reshape(-1, 28, 28)
train_images.shape


# In[ ]:


# 첫번째 이미지 출력
import matplotlib.pyplot as plt
plt.imshow(train_images[0]);


# In[ ]:


# 목표 레이블 
y_train = train.loc[:, 'label']
y_train.unique()


# In[ ]:


# 숫자 레이블을 실제 레이블과 연결하여 확인
target_values = {0 : 'T-shirt/top', 
                 1 : 'Trouser', 
                 2: 'Pullover', 
                 3: 'Dress', 
                 4: 'Coat', 
                 5: 'Sandal', 
                 6: 'Shirt', 
                 7: 'Sneaker', 
                 8: 'Bag', 
                 9: 'Ankle boot'}
print(y_train[0])
print(target_values[y_train[0]])


# In[ ]:


# test 데이터를 28*28 이미지로 변환
test_images = test.loc[:, 'pixel1':].values.reshape(-1, 28, 28)
test_images.shape


# In[ ]:


# 500번째 test 이미지를 출력
plt.imshow(test_images[499]);


# # 2. 데이터 전처리 (Pre-processing)

# In[ ]:


# 피처 스케일 맞추기 
X_train = train_images / 255.
X_test = test_images / 255.
print("최소값:", X_train[0].min())
print("최대값:", X_train[0].max())


# In[ ]:


# 채널 차원 추가
print("변환 전:", X_train.shape, X_test.shape)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
print("변환 후:", X_train.shape, X_test.shape)


# In[ ]:


# Train - Validation 데이터 구분
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val =  train_test_split(X_train, y_train, test_size=0.2, 
                                             stratify=y_train, 
                                             shuffle=True, random_state=SEED)
print("학습 데이터셋 크기: ", X_tr.shape, y_tr.shape)
print("검증 데이터셋 크기: ", X_val.shape, y_val.shape)


# # 3. 모델 구축

# ### MLP 모델
# MLP(Multi-Layer Perceptron)

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
mlp_model = Sequential()
mlp_model.add(Flatten(input_shape=[28, 28]))
mlp_model.add(Dense(units=64, activation='relu'))
mlp_model.add(Dense(units=10, activation='softmax'))

mlp_model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['acc'])

mlp_model.summary()


# In[ ]:


mlp_history = mlp_model.fit(X_tr, y_tr, batch_size=64, epochs=20,
                        validation_data=(X_val, y_val),
                        verbose=2)


# In[ ]:


# 손실 함수 그래프
def plot_loss_curve(history, total_epoch=10, start=1):
    plt.figure(figsize=(5, 5))
    plt.plot(range(start, total_epoch + 1), 
             history.history['loss'][start-1:total_epoch], 
             label='Train')
    plt.plot(range(start, total_epoch + 1), 
             history.history['val_loss'][start-1:total_epoch], 
             label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_loss_curve(history=mlp_history, total_epoch=20, start=1)                   


# ### CNN 활용

# In[ ]:


from tensorflow.keras.layers import Conv2D, MaxPooling2D
cnn_model = Sequential()
cnn_model.add(Conv2D(filters=16, kernel_size=(3, 3), 
                    activation='relu', input_shape=[28, 28, 1]))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(units=64, activation='relu'))
cnn_model.add(Dense(units=10, activation='softmax'))

cnn_model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['acc'])

cnn_model.summary()


# In[ ]:


cnn_history = cnn_model.fit(X_tr, y_tr, batch_size=64, epochs=20,
                        validation_data=(X_val, y_val),
                        verbose=2) 


# In[ ]:


plot_loss_curve(history=cnn_history, total_epoch=20, start=1)    


# In[ ]:


from tensorflow.keras.layers import Dropout
def build_cnn():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3, 3), 
                     activation='relu', input_shape=[28, 28, 1]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=10, activation='softmax'))

    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['acc'])

    return model

cnn_model = build_cnn()
cnn_model.summary()


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss',  patience=10)

cnn_history = cnn_model.fit(X_tr, y_tr, batch_size=64, epochs=100,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping],
                        verbose=0) 


# In[ ]:


# 20 epoch 까지 손실함수와 정확도를 그래프로 나타내기
start=1
end = 20

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].plot(range(start, end+1), cnn_history.history['loss'][start-1:end], 
             label='Train')
axes[0].plot(range(start, end+1), cnn_history.history['val_loss'][start-1:end], 
             label='Validation')
axes[0].set_title('Loss')
axes[0].legend()

axes[1].plot(range(start, end+1), cnn_history.history['acc'][start-1:end], 
             label='Train')
axes[1].plot(range(start, end+1), cnn_history.history['val_acc'][start-1:end], 
             label='Validation')
axes[1].set_title('Accuracy')
axes[1].legend()
plt.show()


# In[ ]:


cnn_model.evaluate(X_val, y_val)


# In[ ]:


y_pred_proba = cnn_model.predict(X_test)
y_pred_classes = np.argmax(y_pred_proba, axis=-1)
y_pred_classes[:10]


# In[ ]:


submission['label'] = y_pred_classes
submission_filepath = drive_path + 'mnist_cnn_submission1.csv'   
submission.to_csv(submission_filepath, index=False)


# In[ ]:


# 사용자 정의 콜백 함수
from tensorflow.keras.callbacks import Callback

class my_callback(Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('val_acc') > 0.91):
      self.model.stop_training = True
      print("\n")
      print("목표 정확도 달성: 검증 정확도 %.4f" % logs.get('val_acc'))

my_callback = my_callback()

# Best Model 저장
from tensorflow.keras.callbacks import ModelCheckpoint

best_model_path = drive_path + "best_cnn_model.h5"
save_best_model = ModelCheckpoint(best_model_path, monitor='val_loss', 
                                  save_best_only=True, save_weights_only=False)

# CNN 모델 학습
cnn_model = build_cnn()
cnn_history = cnn_model.fit(X_tr, y_tr, batch_size=64, epochs=100,
                        validation_data=(X_val, y_val),
                        callbacks=[my_callback, save_best_model],
                        verbose=2) 


# In[ ]:


from tensorflow.keras.models import load_model
# ModelCheckPoint에 저장해둔 모델을 로딩 
best_model = load_model(drive_path + "best_cnn_model.h5")
best_model.summary()


# In[ ]:


y_pred_proba = best_model.predict(X_test)
y_pred_classes = np.argmax(y_pred_proba, axis=-1)
submission['label'] = y_pred_classes
submission_filepath = drive_path + 'mnist_cnn_submission2.csv'   
submission.to_csv(submission_filepath, index=False)