# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 23:48:08 2024

@author: dlgur
"""

#!/usr/bin/env python
# coding: utf-8

# # [07] 인공신경망과 딥 러닝

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

# ### 1-2. 데이터 및 상관관계 확인

# In[1]:


import pandas as pd
w = pd.read_csv("dataset/ch7-1.csv")


# In[2]:

# 데이터 : 300개
# egg_weight : 종란 무게
# acc_food   : 누적 사료량
# weight     : 닭의 몸무게

w.info()


# In[3]:


w.head()


# In[4]:


# 상관분석 실시
w_cor = w.corr(method = 'pearson')
w_cor


# In[5]:


# 상관관계 시각화
import seaborn as sns
sns.pairplot(w)

# 결과:
#   - 종란 무게와 누적 사료량은 닭의 몸무게와 선형 비례    
#   - 특정 구간에서 더 이상 비례하지 않고 꺽임
#   - 종란 무게나 누적 사료량이 많더라도 최종 성장한 닭의 몸무게가 계속 느는 것이 아니다.

#%%

# ### 1-3. 데이터 분할하기

# In[6]:


# 데이터셋 x, y로 분할, ndarray 타입
x_data = w.iloc[:,0:2].values
y_data = w.iloc[:,2].values


# In[8]:


# 데이터 셋 분할을 위한 함수 불러오기
from sklearn.model_selection import train_test_split


# In[9]:


# 훈련용과 테스트용 8:2로 분할 실시
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)


# In[30]:


# 데이터 분할 후 행수 확인
len(pd.DataFrame(x_train)), len(pd.DataFrame(x_test))


#%%

###############################################################################
# ### 1-4. 신경망 구현
###############################################################################

# In[33]:


# MLP 알고리즘 수행을 위한 함수 불러오기
from sklearn.neural_network import MLPRegressor


# In[34]:


# 모델 구축 및 학습
model_mlp = MLPRegressor().fit(x_train, y_train)


# In[35]:


# 모델 파라미터 확인
model_mlp.get_params()


# In[36]:


# 예측값 생성
y_pred_mlp = model_mlp.predict(x_test)


# In[37]:


# 예측값 확인
y_pred_mlp


# ### 1-5. 회귀모델의 성능 평가

# In[41]:


# 회귀성능 지표 계산용 함수 불러오기 및 계산
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# squared = True일 경우 MSE, False일 경우는 RMSE
RMSE = mean_squared_error(y_test, y_pred_mlp, squared = False)
MAE = mean_absolute_error(y_test, y_pred_mlp)
R2 = r2_score(y_test, y_pred_mlp)


# In[42]:


RMSE, MAE, R2

# (144.79788421873667, 119.06608506630535, -31.07103049000726)
# RMSE, MAE 값만으로는 성능을 파악하기 어렵다.
# R2 : -31.07103049000726로 음수 값이 나왔다.
# R²은 1에 가까울수록 뛰어난 성능을 나타낸다.
# R²이 음수라는 것은 오차(실제값 - 예측값) 제곱합이 편차(실제값 - 평균값) 제곱합보다 크다는 의미로 잘못된 예측

#%%

###############################################################################
# ### 1-7. H2O 활용 딥 러닝 구현(회귀)
###############################################################################

# H2O 
# 자바를 기반으로 하는 딥러닝 라이브러리
# DNN만 지원

# In[2]:


# 데이터셋 8:2 분할
from sklearn.model_selection import train_test_split
train, test = train_test_split(w, test_size=0.2)


# In[46]:

# H2O 의존 패키지 설치
# get_ipython().system('pip install requests')
# get_ipython().system('pip install tabulate')
# get_ipython().system('pip install future')
#
# H2O 패키지 설치
# get_ipython().system('pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o')


# In[5]:


# H2O 패키지 실행
import h2o
h2o.init()


# In[8]:


# 훈련용 데이터셋 H2O 전용 데이터프레임으로 변환
hf_train = h2o.H2OFrame(train)


# In[9]:


# 테스트용 데이터셋 H2O 전용 데이터프레임으로 변환
hf_test = h2o.H2OFrame(test)


# In[31]:


hf_train.structure()


# In[32]:


hf_test.structure()


# In[24]:


# 딥러닝 구현을 위한 함수 불러오기
from h2o.estimators import H2ODeepLearningEstimator
# 모델 구축 및 학습
model_h2o = H2ODeepLearningEstimator().train(x=['egg_weight','acc_food'], y='weight', training_frame=hf_train)


# In[25]:


# 모델 정보 확인
model_h2o


# In[26]:


# 모델 성능만 확인
model_h2o.model_performance()


# In[27]:


# 모델 성능지표 R2 확인
model_h2o.r2()


# In[28]:


# 예측값 생성
y_pred_h2o = model_h2o.predict(hf_test)


# In[29]:


y_pred_h2o.head(5)


# In[30]:


# 테스트용 데이터셋에 예측값 predict열로 추가
hf_result = hf_test.cbind(y_pred_h2o)


# In[31]:


hf_result.head(5)


# In[38]:


# sklearn.metrics 모듈 이용을 위해 데이터셋을 h2oframe에서 ndarray로 변경
y_test2 = hf_result[2].as_data_frame().values
y_pred_h2o = hf_result[3].as_data_frame().values


# In[39]:


# 회귀성능 지표 계산용 함수 불러오기 및 계산
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# squared = True일 경우 MSE, False일 경우는 RMSE
RMSE = mean_squared_error(y_test2, y_pred_h2o, squared = False)
MAE = mean_absolute_error(y_test2, y_pred_h2o)
R2 = r2_score(y_test2, y_pred_h2o)


# In[40]:


RMSE, MAE, R2

# (12.250276799326297, 10.005180143628218, 0.7808065378851495)

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
model.add(Input(3))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))


# In[16]:


# 모델 확인
model.summary()


# In[17]:


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


# ### 2-2. 과적합을 줄이는 방법(드롭아웃)

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
