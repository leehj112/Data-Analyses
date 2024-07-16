# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:37:03 2024

@author: leehj
"""

# 모델 읽어 오기
# PART 06 딥러닝 응용
# 이미지 분류: Fashion MNIST 의류 클래스 판별
"""
이미지갯수 : 60,000개
이미지크기 : 784 = 28 * 28
이미지종류 : 0~9까지
"""
#%%

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

# 데이터셋 준비
# 데이콘 사이트에서 다운로드한 CSV파일을 읽어오기
drive_path = "./datasets/"
train = pd.read_csv(drive_path + "fashion_mnist_train.csv")
test = pd.read_csv(drive_path + "fashion_mnist_test.csv")
submission = pd.read_csv(drive_path + "fashion_mnist_submission.csv")

print(train.shape, test.shape, submission.shape)   

#%%
# train 데이터 보기
# label, pixel1, ... pixel784
train.head()

#%%

# train 데이터를 28*28 이미지로 변환
train_images = train.loc[:, 'pixel1':].values.reshape(-1, 28, 28)
train_images.shape # (60000, 28, 28)


#%%
# 첫번째 이미지 출력
import matplotlib.pyplot as plt

fig, axs = plt.subplots(8, 8, figsize=(12,12))
for i in range(8):
    for j in range(8):
        axs[i,j].imshow(train_images[i*8+j])
        axs[i,j].axis('off')
    
plt.show()    


#%%

# 목표 레이블  : 정답
y_train = train.loc[:, 'label']
y_train_unique = y_train.unique()
print(y_train_unique)
print(sorted(y_train_unique)) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#%%

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

#%%

# [문제] 10개의 정답 이미지를 출력


#%%

# test 데이터를 28*28 이미지로 변환
test_images = test.loc[:, 'pixel1':].values.reshape(-1, 28, 28)
test_images.shape


#%%

# 데이터 전처리 (Pre-processing)

# 정규화
# 피처 스케일 맞추기 
# 이미지 픽셀의 값(컬러) : 0~255
X_train = train_images / 255.
X_test = test_images / 255.
print("최소값:", X_train[0].min()) # 0.0
print("최대값:", X_train[0].max()) # 1.0


# In[ ]:

# CNN(합성곱 신경망) : 3채널
# 채널 차원 추가
# 차원 또는 축(axis)를 추가시키려면 np.expand_dim 함수를 사용한다. 
# axis=0 이면 행축 추가
# axis=1 이면 열축 추가
# axis=-1 이면 마지막 축 추가를 의미한다.
print("변환 전:", X_train.shape, X_test.shape) # (60000, 28, 28) (10000, 28, 28)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
print("변환 후:", X_train.shape, X_test.shape) # (60000, 28, 28, 1) (10000, 28, 28, 1)


#%%

###############################################################
# 저장된 모델을 읽어와서 예측을 수행    

from tensorflow.keras.models import load_model
# ModelCheckPoint에 저장해둔 모델을 로딩 
best_model = load_model(drive_path + "best_cnn_model.h5")
best_model.summary()

#%%

y_pred_proba = best_model.predict(X_test)
y_pred_classes = np.argmax(y_pred_proba, axis=-1)
y_pred_classes[:10]

#%%
submission['label'] = y_pred_classes
submission_filepath = drive_path + 'mnist_cnn_submission1.csv'   
submission.to_csv(submission_filepath, index=False)

#%%