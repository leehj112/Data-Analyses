# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 23:45:29 2024

@author: dlgur
"""

#!/usr/bin/env python
# coding: utf-8

# # [06] 분류 및 군집분석

###############################################################################
# ## 2. 병아리의 품종을 구분할 수 있을까? (분류 알고리즘)
###############################################################################

# [분석 스토리]  
# 무럭무럭 잘 자라고 있는 병아리들을 관찰하던 어느 날, 뭔가 특이한 점을 발견했습니다.
# 병아리가 성장함에 따라 생김새가 변하겠지만 병아리들마다 
# 날개 길이, 꽁지깃 길이, 볏의 높이가 유난히 차이가 나 보였습니다. 
# 불안한 마음에 사진을 찍어 종란 판매업체 담당자에게 이 병아리들이 같은 품종이 맞는지 문의했습니다. 
# 그리고 담당자에게서 온 답변은 매우 혼란스럽게 만들었습니다. 
# 종란 판매 직원의 실수로 주문을 넣었던 A 품종의 종란뿐만 
# 아니라 B와 C라는 2가지 품종의 종란이 섞여서 납품되었다는 것입니다. 
# 졸지에 3가지 품종의 병아리를 키우게 되어 판매처에 클레임(claim)을 제기했고, 
# 종란 판매처의 품종 엔지니어가 양계농장에 급파되었습니다. 
# 엔지니어는 하루에 걸쳐 총 300마리의 병아리 품종을 정확히 구분해 기록했습니다. 
# 다음에도 혹시나 이런 일이 발생할지 모른다는 불안감에 
# 이 300마리의 병아리 데이터를 활용해 품종을 구분할 수 있는 분류 모델을 개발해 보려고 합니다. 
# 과연 그는 암수를 구분했던 것처럼 품종도 잘 구분해낼 수 있을까요?

# In[21]:

# 실습용 데이터
# 지도학습 사례(Supervised Learning)    
# 병아리 300마리 품종 데이터를 훈련 데이터 테스트 용도의 2가지 데이터 셋으로 8:2 비율로 분할

# 실습용 데이터 불러오기
# wing_length : 날개 길이
# tail_length : 꽁지깃 길이
# comb_height : 볏 높이
# breeds      : 품종('a', 'b', 'c')

import pandas as pd
df_train = pd.read_csv('dataset/ch6-2_train.csv')
df_test = pd.read_csv('dataset/ch6-2_test.csv')


# In[22]:

# 훈련 데이터: 240건
df_train.info()


# In[23]:

# 테스트 데이터: 60건
df_test.info()


# In[24]:


df_train.head()

#%%

df_train['breeds'].value_counts() # a:80, b:80, c:80

#%%

df_test['breeds'].value_counts() # a:20, b:20, c:20


# In[25]:


# train, test 데이터셋 각각 x, y로 분할, ndarray 타입
x_train = df_train.iloc[:,0:3].values
y_train = df_train.iloc[:,3].values
x_test = df_test.iloc[:,0:3].values
y_test = df_test.iloc[:,3].values

#%%

###############################################################################
# ### 2-5. 배깅(Bagging)
###############################################################################

# 배깅(Bagging)은 앙상블(Ensemble) 모형 중 하나입니다. 
# 앙상블은 프랑스어로 “통일” 또는 “조화”를 의미하는 용어입니다. 
# 이런 의미처럼 앙상블 모형은 여러 개의 예측 모델을 만든 후 조합해 하나의 최적화된 최종 예측 모델을 만듭니다. 
# 앙상블 모형은 분류와 회귀 모두에 사용할 수 있는 알고리즘입니다.  
# 앙상블 모형에는 배깅과 부스팅(Boosting)이 있습니다. 
#
# 배깅은 Bootstrap Aggregating의 줄임말로 학습 데이터 셋으로부터 동일한 크기의 표본을 단순 랜덤 복원 추출해 여러 개 만들고, 
# 각 표본에 대한 예측 모델을 생성한 후 결합해 최종 예측 모델을 만드는 알고리즘입니다. 
# 학습 데이터 셋에서 단순 랜덤 복원 추출해 동일한 크기의 표본을 여러 개 만드는 
# 샘플링 방법을 부트스트랩(Bootstrap)이라고 합니다.

# 앙상블 모형의 종류 및 발달 과정
#   - 의사결정나무(Decision Tree) -> 랜덤포레스트(Random Forest)
#   - 앙상블(Ensemble)
#     - 배깅(Bagging, Bootstrap Aggregation) ->  랜덤포레스트(Random Forest)
#     - 부스팅(Boosting)
#       - AdaBoost(Adaptive Boosting)
#       - GBM(Gradient Boosting Machine)
#         - XGBoost(eXtreme Gradient Boosting)
#         - Light GBM
#   

#%%

###############################################################################
# ### 2-9. XGBoost
###############################################################################

# XGBoost의 특징
#   - Gradient Boosting 알고리즘을 기반으로 사용
#   - 병렬 처리를 기반으로 하기 때문에 속도가 빠르며 과적합 규제 기능
#   - 분류(classification)와 회귀(regression) 모두 적용 가능
#   - 특정 기준에 맞으면 지정한 학습 횟수에 도달하기 전에 학습을 종료시키는 조기종료(Early Stopping)

#%%
# ① XGBoost

# In[85]:


# xgboost 패키지 설치(anaconda 기본 패키지가 아님)
# 주피터: get_ipython().system('pip install xgboost')
# 일반환경: pip install xgboost


# In[86]:


# XGBoost 알고리즘 수행을 위한 함수 불러오기
from xgboost import XGBClassifier


# In[87]:

# LabelEncoder 객체 생성

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

# 범주형 데이터를 수치형으로 변환
y_train = label_encoder.fit_transform(df_train.iloc[:,3].values)    
y_test = label_encoder.fit_transform(df_test.iloc[:,3].values)

#%%

# 모델 구축 및 학습
model_xgb = XGBClassifier().fit(x_train, y_train)


# In[88]:


y_pred_xgb = model_xgb.predict(x_test)


# In[89]:


from sklearn.metrics import confusion_matrix
# 위쪽이 예측값, 좌측이 실제값
confusion_matrix(y_test, y_pred_xgb)

#%%

"""
array([[20,  0,  0],
       [ 1, 18,  1],
       [ 0,  1, 19]], dtype=int64)
"""

# In[90]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_xgb))

#%%

# 정확도 : 0.95

"""
              precision    recall  f1-score   support

           0       0.95      1.00      0.98        20
           1       0.95      0.90      0.92        20
           2       0.95      0.95      0.95        20

    accuracy                           0.95        60
   macro avg       0.95      0.95      0.95        60
weighted avg       0.95      0.95      0.95        60
"""


#%%

###############################################################################
# ② 그리드서치를 이용한 하이퍼 파라미터 튜닝
###############################################################################

# 하이퍼 파라미터(Hyper Parameter)
#   - 학습률(eta)
#   - 트리깊이(max_depth)
#   - gamma
#   - lambda
#   - alpha

# In[91]:


# 그리드서치 함수 불러오기
from sklearn.model_selection import GridSearchCV


# In[92]:


# 그리드서치로 실행할 하이퍼 파라미터 딕셔너리 타입으로 저장
xgb_param_grid = {
    'eta' : [0.05, 0.1, 0.3, 0.5],
    'gamma' : [0, 0.5, 1, 2],
    'max_depth' : [2, 4, 8, 12]
}


# In[93]:


# 평가기준이 정확도인 그리드서치 모델 구축(n_jobs=-1은 모든 CPU 코어를 사용하라는 의미) 
xgb_grid = GridSearchCV(XGBClassifier(), param_grid = xgb_param_grid, n_jobs=-1, scoring = 'accuracy')


# In[94]:


# 그리드서치 모델 학습
xgb_grid.fit(x_train, y_train)


# In[95]:


# 정확도 최고 점수
xgb_grid.best_score_

# 결과: 0.95

# In[96]:


# 정확도 최고일 때 하이퍼 파라미터
xgb_grid.best_params_

# {'eta': 0.1, 'gamma': 0, 'max_depth': 8}


# In[97]:


# 선정된 하이퍼 파라미터로 재학습
model_xgb2 = XGBClassifier(eta=0.1, gamma =0, max_depth=8).fit(x_train, y_train)


# In[98]:


y_pred_xgb2 = model_xgb2.predict(x_test)


# In[99]:


from sklearn.metrics import confusion_matrix
# 위쪽이 예측값, 좌측이 실제값
confusion_matrix(y_test, y_pred_xgb2)


# In[100]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_xgb2))


#%%

###############################################################################
# ## 3. 효과적인 사육을 위해 사육환경을 분리해보자 (군집 알고리즘)
###############################################################################

# 군집(clustering)
#   - 비지도학습(Unsupervised Learning)
#   - 여러 개의 독립변수들을 활용해 유사한 특징을 갖는 개체들을 몇 개의 군집으로 집단화시키는 방법
#   - 종속변수(라벨)가 없다는 점에서 분류 알고리즘과 구분
#   - 군집 알고리즘은 계층적(Hierarchical) 방법과 비계층적(non-hierarchical) 방법으로 나눔

#
# 계층적(Hierarchical) 방법
#   - 가장 거리가 가가운 개체들을 결합해 나가는 과정을 반복해 원하는 개수의 군집을 형성해 나가는 방법

# 개체간 거리를 계산하는 방법
#   - 거리 계산법은 독립변수가 연속형
#   - 유클리디안 거리
#   - 맨해튼 거리
#   - 민코프스키 거리
#   - 표준화 거리
#   - 마할라노비스 거리
#   - 체비세프 거리
#   - 캔버라 거리   

# 독립변수가 범주형일 경우
#   - 자카드(Jaccard) 계수(유사도)를 이용

# 
# 군집 간의 연결하는 방법
#   - 단일 연결법
#   - 완전 연결법
#   - 평균 연결법
#   - 중심 연결법
#   - 메디안 연결법 
#   - 와드 연결법

# In[101]:


import pandas as pd
cl = pd.read_csv("dataset/ch6-3.csv")


# In[102]:


cl.info()


# In[103]:


cl.head()


# In[104]:


cl.describe()


# In[105]:

# 산점도
# 하루 평균 사료 섭취량(food)에 다른 몸무게(weight) 데이터의 분포
import matplotlib.pyplot as plt
plt.figure(figsize = (10,7))
plt.scatter(x = cl['food'], y = cl['weight'])
plt.show()


# In[106]:

###############################################################################    
# k-평균 군집(k-Means Clustering)
###############################################################################    

# 실행과정
#   1. 초기 k 평균값은 데이터 개체 중에서 랜덤하게 선택된다.
#   2. k를 중심으로 각 데이터의 개체들은 가장 가까이 있는 평균값을 기준으로 묶인다.
#   3. k개 군집의 각 중심점을 기준으로 평균값이 재조정된다.
#   4. k개 군집의 각 중심점과 각 군집 내 개체들 간의 거리의 제곱합이 최소가 될 때까지
#      위 2번과 3번의 과정을 반복한다.

#%%    

# K-Measn 군집 알고리즘 수행을 위한 함수 불러오기
from sklearn.cluster import KMeans


# In[107]:


cl_n = cl.iloc[:,1:3].values


# In[108]:


# 연속형 데이터만으로 군집 실행
cl_kmc = KMeans(n_clusters = 3).fit(cl_n)


# In[109]:


# 군집결과 확인
cl_kmc.labels_


# In[110]:


# 군집별 갯수 확인
import numpy as np
np.unique(cl_kmc.labels_, return_counts = True)

# 군집: 3개, 0,1,2
# (array([0, 1, 2]), array([33, 32, 35], dtype=int64))

# In[111]:

# 산점도
# 3개의 그룹이 하루 평균 사료 섭취량과 몸무게에 따라 나눠짐
plt.figure(figsize = (10,7))
plt.scatter(x = cl['food'], y = cl['weight'], c = cl_kmc.labels_)
plt.show()

#%%

# THE END