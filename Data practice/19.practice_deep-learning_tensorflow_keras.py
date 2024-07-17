# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 23:49:16 2024

@author: dlgur
"""

#!/usr/bin/env python
# coding: utf-8

# # [07] 인공신경망과 딥 러닝

"""
# 문제 종류에 따른 일반적인 하이퍼 파라미터 설정값
------------+---------------------+--------------+--------------+---------------------
   단계     |  하이퍼 파라미터    | 이진분류     | 다중분류     | 회귀
------------+---------------------+--------------+--------------+---------------------
 신경망구축 |  출력층 활성화 함수 | sigmoid      |  softmax     | linear 또는 생략
------------+---------------------+--------------+--------------+---------------------
            |  손실함수(loss)     | binary       | categorical  | mse, mae
 학습설정   |                     | crossentropy | crossentropy |
(compile)   +---------------------+--------------+--------------+---------------------
            |  평가지표(metrics)  | accuracy     | accuracy     | mae, mse, rmse
------------+---------------------+--------------+--------------+---------------------

"""

# 과적합 :
#   - 학습한 결과보다 테스트한 결과의 정확도가 확연히 낮거나 오차가 크게 나오는 경우

# 딥러닝에서 과적합을 줄이는 방법
#   - 데이터의 양 늘리기(충분한 학습과 검증 환경 조성)
#   - 모델의 복잡도 줄이기(은익층 수 및 가중치 규제)
#   - 드롭아웃(특정 노드 과대 의존 방지)


#%%


# [분석스토리]  
# 어느덧 병아리가 부화하고, 성장한 지 70일이 지났습니다. 
# 이제 어엿한 닭으로 성장했지만 타사 대비 닭의 발육 상태가 떨어진 것을 확인했습니다. 
# 이에 그 원인을 분석하기 시작했습니다. 
# 그리고 도출한 결론은 종란 무게와 닭으로 성장할 때까지의 누적 사료량 관리가 가장 중요한 변수라고 판단했습니다. 
# 따라서 앞으로 체계적인 닭의 발육관리를 위해 종란 무게 및 누적 사료량에 따른 닭의 몸무게 예측 모델을 개발하기로 했습니다.

# ## 1. 성장한 닭의 체중을 예측할 수 있을까? (회귀)

# ### 1-1. 인공신경망이란?

# 우리 몸의 신경세포(Neuron)는 수상 돌기(Dendrite)를 통해 전기적 신호를 입력받아 신호의 강도가 일정 기준을 초과할 때 
# 즉, 역치 이상의 자극이 주어질 때 신경세포가 활성화되어 축삭 돌기(Axon)를 통해 신호를 내보냅니다. 
# 이런 중추신경계를 모방하여 만들어진 머신러닝 알고리즘이 인공신경망(Artificial Neural Network)입니다.
# <div>
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/Neuron_Hand-tuned.svg/400px-Neuron_Hand-tuned.svg.png" width="500"/>
#     <center>신경세포(출처 : wikipedia)</center>
# </div>

#%%

# ## 2. 딥 러닝을 이용해 병아리의 품종을 다시 구분해 보자! (분류)

# ### 2-1. Keras 활용 딥 러닝 구현(분류)

# In[1]:


# tensorflow & keras 설치
# get_ipython().system('pip install tensorflow')
# get_ipython().system('pip install keras')


# In[2]:


# 딥 러닝용 함수 불러오기
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Input, Dense


# In[3]:


# 실습용 데이터 불러오기
import pandas as pd
df_train = pd.read_csv('dataset/ch6-2_train.csv')
df_test = pd.read_csv('dataset/ch6-2_test.csv')

# In[4]:


# train, test 데이터셋 각각 x, y로 분할, ndarray 타입
x_train = df_train.iloc[:,0:3].values
y_train = df_train.iloc[:,3].values
x_test = df_test.iloc[:,0:3].values
y_test = df_test.iloc[:,3].values


# In[5]:


# 독립변수 분포 확인
df_test.describe()


# In[6]:


# 히스토그램으로 독립변수 분포 확인
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 10))
plt.hist(df_train.wing_length, label = 'wing_length')
plt.hist(df_train.tail_length, label = 'tail_length')
plt.hist(df_train.comb_height, label = 'comb_height')
plt.legend()
plt.show()

#%%

# 결과:
# 3가지 변수 모두 데이터의 중심이 좌측이나 우측으로 치우치치 않고, 주로 가운데 모여있는 형태
# 이런 형태는 Min-Max Scaler를 통해 최솟값 0, 최대값 1로 바꾸는 것이 좋음
# 만약 데이터의 중심이 한 쪽에 치우친 형태라면 Standard 또는 Robust Scaler를 이용해
# 스케일된 데이터의 중심이 가운데로 향하게 바꿔주는 것이 모델의 성능 향상에 유리

# In[7]:


# Min-Max Scaling을 위한 함수 불러오기
from sklearn.preprocessing import MinMaxScaler

# x_train 데이터로 학습한 Min-Max Scaler 생성
mms = MinMaxScaler().fit(x_train)


# In[8]:


# 스케일링 실시
mm_x_train = mms.transform(x_train)
mm_x_test = mms.transform(x_test)


# In[9]:


# 스케일링 결과 확인
mm_x_train[0:5]


# In[10]:


# 최솟값, 최댓값 확인
mm_x_train.min(), mm_x_train.max()


# In[11]:


# 종속변수 확인
y_train[0:5]


# In[12]:


# One-Hot Encoding을 위한 함수 불러오기
from sklearn.preprocessing import LabelBinarizer

# y_train 데이터로 학습한 LabelBinarizer 생성
lb = LabelBinarizer().fit(y_train)


# In[13]:


# 종속변수 a, b, c를 이진수로 One-Hot Encoding
o_y_train = lb.transform(y_train)
o_y_test = lb.transform(y_test)


# In[14]:


# One-Hot Encoding 결과 확인
o_y_train[0:5]


# In[15]:


# 모델 구축
model = Sequential()
model.add(Input(3)) # 입력층: 독립변수가 3개
model.add(Dense(16, activation = 'relu')) # 은닉층
model.add(Dense(16, activation = 'relu')) # 은닉층
model.add(Dense(3, activation = 'softmax')) # 출력층: 종속변수 라벨 갯수 3개


# In[16]:


# 모델 확인
model.summary()


# In[17]:

# 손실함수 : 모델의 최적화에 사용되는 목적 함수
#   - 이진분류: binary_crossentropy
#   - 다중분류: categorical_crossentropy
#   - 회귀: mse, mae
# mertics: 평가지표
#   - 분류: accuracy
#   - 회귀: mse, mae, rmse
    

# 모델 학습 설정(compile)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = 'accuracy')


# In[18]:


# 모델 학습(fit) 실시
history = model.fit(mm_x_train, o_y_train, epochs = 300, batch_size = 16, validation_split = 0.2)


# In[20]:


# 학습 결과 그래프 표시
plt.figure(figsize = (16,5))
plt.subplot(1,2,1) # 그래프 좌측 표시
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(history.history['loss'], label = 'Train Loss')
plt.plot(history.history['val_loss'], label = 'Val Loss')
plt.legend()
plt.subplot(1,2,2) # 그래프 우측 표시
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(history.history['accuracy'], label = 'Train Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Val Accuracy')
plt.legend()
plt.show()


# In[21]:


# 모델 훈련성능(loss, accuracy) 확인
model.evaluate(mm_x_train, o_y_train)


# In[45]:


# 모델 테스트성능(loss, accuracy) 확인
model.evaluate(mm_x_test, o_y_test)


# In[48]:


# 테스트용 데이터셋 활용 예측값 생성
y_pred = model.predict(mm_x_test)


# In[49]:


# 예측값 확인
y_pred[0:5]


# In[50]:


# 예측값 반올림
y_pred = y_pred.round()


# In[51]:


# 예측값 확인
y_pred[0:5]


# In[52]:


import numpy as np
# 배열에서 열 기준 가장 큰 값의 인덱스 호출 및 배열 차원 축소
y_pred = np.argmax(y_pred, axis=1).reshape(-1)


# In[53]:


# 변환된 예측값 확인
y_pred


# In[56]:


# 예측값 배열의 0을 a로, 1을 b로, 2를 c로 변경
condlist = [y_pred == 0, y_pred == 1, y_pred == 2]
choicelist = ['a', 'b', 'c']
y_pred = np.select(condlist, choicelist)


# In[57]:


# 변환된 예측값과 실제값 확인
print(y_pred[0:5], y_test[0:5])


# In[32]:


from sklearn.metrics import confusion_matrix
# 위쪽이 예측값, 좌측이 실제값
confusion_matrix(y_test, y_pred)


# In[33]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


###############################################################################
# ### 2-2. 과적합을 줄이는 방법(드롭아웃)
###############################################################################

# 딥러닝에서 과적합을 줄이는 방법
#   - 데이터의 양 늘리기(충분한 학습과 검증 환경 조성)
#   - 모델의 복잡도 줄이기(은익층 수 및 가중치 규제)
#   - 드롭아웃(특정 노드 과대 의존 방지)

# In[34]:


from keras.layers import Dropout


# In[35]:


# 모델 구축
model_d = Sequential()
model_d.add(Input(3))
model_d.add(Dense(16, activation = 'relu'))
model_d.add(Dropout(0.4))
model_d.add(Dense(16, activation = 'relu'))
model_d.add(Dropout(0.4))
model_d.add(Dense(3, activation = 'softmax'))


# In[36]:


model_d.summary()
