# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:35:20 2024

@author: leehj
"""

# PART 06 딥러닝 응용
# 이미지 분류: Fashion MNIST 의류 클래스 판별
# MLP(Multi-Layer Perceptron)

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
"""
이미지 갯수 : 60,000개
이미지 크기 : 784 = 28 * 28
이미지 종류 : 0~9까지(10가지 종류)
"""

# 데이콘 사이트에서 다운로드한 CSV파일을 읽어오기
drive_path = "./datasets/"
train = pd.read_csv(drive_path + "fashion_mnist_train.csv")
test = pd.read_csv(drive_path + "fashion_mnist_test.csv")
submission = pd.read_csv(drive_path + "fashion_mnist_submission.csv")

print(train.shape, test.shape, submission.shape)   
# 훈련, 검증, 제출
# (60000, 786) (10000, 785) (10000, 2)


# In[ ]:


# train 데이터 보기
# label, pixel1, ... pixel784
train.head()


# In[ ]:


# train 데이터를 28*28 이미지로 변환
# 컬럼: 'pixel1':'pixel784
train_images = train.loc[:, 'pixel1':].values.reshape(-1, 28, 28)
train_images.shape # (60000, 28, 28)


# In[ ]:


# 첫번째 이미지 출력
import matplotlib.pyplot as plt
plt.imshow(train_images[0])
plt.imshow(train_images[0], cmap='gray')    # 훈련을 위해 변환된 이미지 형태
plt.imshow(train_images[0], cmap='gray_r')  # 반전: 원래 이미지

print("train_images.shape:", train_images.shape)

#%%

imglen = train_images.shape[0]
print("이미지 갯수: ", imglen)  # 60000

#%%

fig, axs = plt.subplots(8, 8, figsize=(12,12))
for i in range(8):
    for j in range(8):
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
target_values = {0 : 'T-shirt/top', # T셔츠
                 1 : 'Trouser',     # 바지
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


###############################################################################
# # 2. 데이터 전처리 (Pre-processing)

# In[ ]:

# 정규화
# 피처 스케일 맞추기 
# 이미지 픽셀의 값(컬러) : 0~255
# 픽셀이 0.0 ~ 1.0 범위로 축소
X_train = train_images / 255.
X_test = test_images / 255.
print("최소값:", X_train[0].min()) # 0.0
print("최대값:", X_train[0].max()) # 1.0


# In[ ]:

# CNN(합성곱 신경망) : 3채널(Red, Green, Blue)
# 채널 차원 추가: 컬러 채널
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
print("학습 데이터셋 크기: ", X_tr.shape, y_tr.shape)   # (48000, 28, 28, 1) (48000,)
print("검증 데이터셋 크기: ", X_val.shape, y_val.shape) # (12000, 28, 28, 1) (12000,)


#%%
# 3. 모델 구축
# MLP 모델

#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

#%%

mlp_model = Sequential()

# 입력크기 지정
mlp_model.add(Flatten(input_shape=[28, 28])) # 입력층: 1차원 배열로 펼침
print(mlp_model.output_shape) # (None, 784)

mlp_model.add(Dense(units=64, activation='relu'))    # 은닉층
mlp_model.add(Dense(units=10, activation='softmax')) # 출력층

# loss: sparse_categorical_crossentropy
# 
mlp_model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['acc'])

mlp_model.summary()

#%%
"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 flatten (Flatten)           (None, 784)               0         
                                                                 
 dense (Dense)               (None, 64)                50240     
                                                                 
 dense_1 (Dense)             (None, 10)                650       
                                                                 
=================================================================
Total params: 50890 (198.79 KB)
Trainable params: 50890 (198.79 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
"""

# In[ ]:


mlp_history = mlp_model.fit(X_tr, y_tr, batch_size=64, epochs=20,
                        validation_data=(X_val, y_val),
                        verbose=2)


# In[ ]:

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

plot_loss_curve(history=mlp_history, total_epoch=20, start=1)                   

# 결과: 과대적합
# 훈련 오차는 계속 감소하지만 검증 오차는 0.34 수준에서 횡보하는 추세
# 3번째 에포크부터 훈련 오차가 검증 오차보다 낮아지기 시작하여 과대적합이 발생

#%%

# 평가
mlp_model.evaluate(X_val, y_val) # [0.3510977029800415, 0.8818333148956299]
# val_loss : 0.3510977029800415
# val_acc: 0.8818333148956299

#%%