# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 23:48:50 2024

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


# ### 1-4. 신경망 구현

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

# THE END