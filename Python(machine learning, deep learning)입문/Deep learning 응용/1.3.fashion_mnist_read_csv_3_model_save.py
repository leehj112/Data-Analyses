# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:36:20 2024

@author: leehj
"""

# 모델 저장하기
# PART 06 딥러닝 응용
# 이미지 분류: Fashion MNIST 의류 클래스 판별
"""
이미지갯수 : 60,000개
이미지크기 : 784 = 28 * 28
이미지종류 : 0~9까지
"""

# In[ ]:


# 라이브러리 설정
import pandas as pd
import numpy as np
import tensorflow as tf
import random

#%%
# 랜덤 시드 고정
SEED=12
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)  

#%%

# # 2. 데이터셋 준비

# In[ ]:

# 데이콘 사이트에서 다운로드한 CSV파일을 읽어오기
"""
이미지 갯수 : 60,000개
이미지 크기 : 784 = 28 * 28
이미지 종류 : 0~9까지(10가지 종류)
"""
drive_path = "./datasets/"
train = pd.read_csv(drive_path + "fashion_mnist_train.csv")
test = pd.read_csv(drive_path + "fashion_mnist_test.csv")
submission = pd.read_csv(drive_path + "fashion_mnist_submission.csv")

print(train.shape, test.shape, submission.shape) # (60000, 786) (10000, 785) (10000, 2)

# In[ ]:


# train 데이터 보기
# label, pixel1, ... pixel784
train.head()


# In[ ]:


# train 데이터를 28*28 이미지로 변환
train_images = train.loc[:, 'pixel1':].values.reshape(-1, 28, 28)
train_images.shape # (60000, 28, 28)


# In[ ]:


# 첫번째 이미지 출력
import matplotlib.pyplot as plt
plt.imshow(train_images[0]);

print("train_images.shape:", train_images.shape) # (60000, 28, 28)

#%%

imglen = train_images.shape[0]
print("이미지 갯수: ", imglen)    # 60000

#%%

# 64개 훈련 이미지 출력
fig, axs = plt.subplots(8, 8, figsize=(12,12))
for i in range(8):
    for j in range(8):
        # axs[i,j].imshow(train_images[i*8+j])
        axs[i,j].imshow(train_images[i*8+j], cmap='gray_r')
        axs[i,j].axis('off')
    
plt.show()    


# In[ ]:


# 목표 레이블  : 정답
y_train = train.loc[:, 'label']
y_train_unique = y_train.unique()
print(y_train_unique)
print(sorted(y_train_unique)) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]



# In[ ]:


# 숫자 레이블을 실제 레이블과 연결하여 확인
target_values = {0: 'T-shirt/top',  # T셔츠
                 1: 'Trouser',      # 바지
                 2: 'Pullover',     # 스웨터(소매달린)
                 3: 'Dress',        # 드레스
                 4: 'Coat',         # 코드
                 5: 'Sandal',       # 샌달
                 6: 'Shirt',        # 셔츠
                 7: 'Sneaker',      # 운동화
                 8: 'Bag',          # 백
                 9: 'Ankle boot'}   # 부츠
print(y_train[0])
print(target_values[y_train[0]])

#%%

# test 데이터를 28*28 이미지로 변환
test_images = test.loc[:, 'pixel1':].values.reshape(-1, 28, 28)
test_images.shape # (10000, 28, 28)


# In[ ]:


# 500번째 test 이미지를 출력
plt.imshow(test_images[499]);
plt.imshow(test_images[499], cmap='gray');
plt.imshow(test_images[499], cmap='gray_r');


#%%

# 2. 데이터 전처리 (Pre-processing)

# 정규화
# 피처 스케일 맞추기 
# 이미지 픽셀의 값(컬러) : 0~255
X_train = train_images / 255.
X_test = test_images / 255.
print("최소값:", X_train[0].min()) # 0.0
print("최대값:", X_train[0].max()) # 1.0


# In[ ]:

# CNN(합성곱 신경망) : 3채널(컬러:Red, Green, Blue)
# 채널 차원 추가
# 차원 또는 축(axis)를 추가시키려면 np.expand_dim 함수를 사용한다. 
# axis=0 이면 행축 추가
# axis=1 이면 열축 추가
# axis=-1 이면 마지막 축 추가를 의미한다.
print("변환 전:", X_train.shape, X_test.shape) # (60000, 28, 28) (10000, 28, 28)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
print("변환 후:", X_train.shape, X_test.shape) # (60000, 28, 28, 1) (10000, 28, 28, 1)


# In[ ]:

# 폴드아웃 교차검증
# Train - Validation 데이터 구분
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val =  train_test_split(X_train, y_train, test_size=0.2, 
                                             stratify=y_train, 
                                             shuffle=True, random_state=SEED)
print("학습 데이터셋 크기: ", X_tr.shape, y_tr.shape)
print("검증 데이터셋 크기: ", X_val.shape, y_val.shape)


#%%
# 3. 모델 구축

#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout

#%%
# History 객체. 
# History.history 속성은 연속된 세대에 걸친 학습 손실 값과 측정항목 값,
# 그리고 (적용 가능한 경우) 검증 손실 값과 검증 측정항목 값의 기록입니다.

# 손실 함수 그래프
def plot_loss_curve(history, total_epoch=10, start=1):
    plt.figure(figsize=(5, 5))
    plt.plot(range(start, total_epoch + 1), 
             history.history['loss'][start-1:total_epoch], # 손실값
             label='Train')
    plt.plot(range(start, total_epoch + 1), 
             history.history['val_loss'][start-1:total_epoch], # 검증손실값
             label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


#%%

# 합성곱 신경망(CNN, Convolutional Neural Network)
# 드롭아웃: 
#   - 모델을 가볍게 해서 과대적합 해소를 위해서
#   - 지정된 비율 만큼 뉴런 가중치를 0으로 만듦 
def build_cnn():
    model = Sequential()
    model.add(Conv2D(filters=16,         # 필터(뉴런)
                     kernel_size=(3, 3), # 커널크기(3*3)
                     activation='relu', input_shape=[28, 28, 1]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(rate=0.5)) # 드롭아웃
    model.add(Dense(units=10, activation='softmax')) # 다중분류: 전체 비율의 합이 1.0

    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', # 원핫 인코딩을 하지 않아도 된다.
                metrics=['acc'])

    return model

#%%
cnn_model = build_cnn()
cnn_model.summary()


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss',  patience=10)

cnn_history = cnn_model.fit(X_tr, y_tr, batch_size=64, epochs=10,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping],
                        verbose=2) 


# In[ ]:


# 20 epoch 까지 손실함수와 정확도를 그래프로 나타내기
start=1
end = 10

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


# 평가
cnn_model.evaluate(X_val, y_val)
# [0.33007949590682983, 0.9149166941642761]

# In[ ]:

# 테스트 데이터 10000개의 예측결과    
# y_pred_proba : (10000, 10)
y_pred_proba = cnn_model.predict(X_test)

# 예측된 결과중에서 가장 큰 값을 가진 컬럼의 인덱스를 구함
# 인덱스: 예측한 의류의 종류(0~9)
# y_pred_classes : (10000,)
y_pred_classes = np.argmax(y_pred_proba, axis=-1) 

y_pred_classes[:10] # array([0, 1, 2, 6, 3, 6, 8, 6, 5, 0], dtype=int64)


# In[ ]:


submission['label'] = y_pred_classes
submission_filepath = drive_path + 'mnist_cnn_submission1.csv'   
submission.to_csv(submission_filepath, index=False)


#%%

# 사용자 정의 콜백 함수
from tensorflow.keras.callbacks import Callback

# Callback 클래스를 상속
# on_epoch_end : 메소드 재정의
#   - 매 애포크가 끝날 때마다 실행
#   - 종료 조건을 체크
# model.stop_training 속성값 : 정확도가 91%보다 크면 종료(훈련을 멈춤)
class my_callback(Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('val_acc') > 0.91):
      self.model.stop_training = True
      print("\n")
      print("목표 정확도 달성: 검증 정확도 %.4f" % logs.get('val_acc'))

my_callback = my_callback()

#%%
# Best Model 저장
from tensorflow.keras.callbacks import ModelCheckpoint

# 훈련된 모델을 저장
# 파일형식: HDF5(*.h5)
# monitor: 'val_loss', 
#   - 손실 함수를 기준으로 모델의 학습 상태를 추적
#   - 매 에포크마다 모델을 저장
# save_best_only: True, 모델이 기존 최고치보다 높은 성능을 발휘할 때만 저장(가장 높은 성능)
# ave_weights_only: False, 모델 아키텍처와 가중치를 함께 저장
best_model_path = drive_path + "best_cnn_model.h5"
save_best_model = ModelCheckpoint(best_model_path, monitor='val_loss', 
                                  save_best_only=True, save_weights_only=False)

# CNN 모델 학습
cnn_model = build_cnn()
cnn_history = cnn_model.fit(X_tr, y_tr, batch_size=64, epochs=100,
                        validation_data=(X_val, y_val),
                        callbacks=[my_callback, save_best_model],
                        verbose=2) 


#%%

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