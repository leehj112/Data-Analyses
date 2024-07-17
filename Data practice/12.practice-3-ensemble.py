# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 23:45:00 2024

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

# In[56]:

# 배깅(Bagging, Bootstrap Aggregation)
# 학습 데이터 셋으로부터 동일한 크기의 표본을 단순 랜덤 복원 추출해 여러 개 만들고,
# 각 표본에 대한 예측 모델을 생성한 뒤 결합해 최종 예측 모델을 만드는 알고리즘
# 여기서 학습 데이터 겟에서 단순 랜덤 복원 추출해 동일한 크기의 표본을 여러 개 만드는
# 샘플링 방법을 부트스트랩(Bootstrap)이라고 한다.

# 배깅 알고리즘 수행을 위한 함수 불러오기
from sklearn.ensemble import BaggingClassifier


# In[57]:


# 모델 구축 및 학습
model_bag = BaggingClassifier().fit(x_train, y_train)


# In[58]:


y_pred_bag = model_bag.predict(x_test)


# In[59]:


from sklearn.metrics import confusion_matrix
# 위쪽이 예측값, 좌측이 실제값
confusion_matrix(y_test, y_pred_bag)


# In[60]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_bag))


#%%

###############################################################################
# ### 2-6. 부스트
###############################################################################

# 부스팅(Boosting)은 앙상블 모형 중 하나로 배깅이 부트스트랩 시 각 표본에 동일한 확률을 부여하는 것과 달리 
# 잘못 분류된 표본에 더 큰 가중치를 적용해 새로운 분류 규칙을 만들고, 
# 이런 과정을 반복해 최종 모형을 만드는 알고리즘입니다. 
# 부스팅은 최근까지도 AdaBoost(Adaptive Boosting), GBM(Gradient Boosting Machine)과 같은 알고리즘이 나오면서 
# 배깅보다 성능이 뛰어난 경우가 많습니다. 
# 특히 XGBoost의 경우 캐글(Kaggle)에서 상위 랭커들이 사용해 높은 인기를 얻은 알고리즘입니다.

# ① 에이다부스트

# In[61]:


# 에이다부스트 알고리즘 수행을 위한 함수 불러오기
from sklearn.ensemble import AdaBoostClassifier


# In[62]:


# 모델 구축 및 학습
model_adb = AdaBoostClassifier().fit(x_train, y_train)


# In[63]:


y_pred_adb = model_adb.predict(x_test)


# In[64]:


from sklearn.metrics import confusion_matrix
# 위쪽이 예측값, 좌측이 실제값
confusion_matrix(y_test, y_pred_adb)


# In[65]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_adb))

# 결과 : 정확도(0.92)


#%%

###############################################################################
# ② 그래디언트부스트
###############################################################################

# Gradient Boosting Machine(GBM)
# Gradient Descent를 이용해 순차적으로 틀린 것에 가중치를 부여해 보다 나은 모델을 만드는 부스팅 알고리즘이다.
# Gradient Descent는 '경사 하강법'이라고 한다.
# 함수의 기울기(경사)를 구하고, 기울기의 절대값을 낮은 쪽으로 계속 이동시켜 함수가 최솟값(기울기 - 0)이 될 때까지 반복한다.

# In[66]:


# 그래디언트부스트 알고리즘 수행을 위한 함수 불러오기
from sklearn.ensemble import GradientBoostingClassifier


# In[67]:


# 모델 구축 및 학습
model_gb = GradientBoostingClassifier().fit(x_train, y_train)


# In[68]:


y_pred_gb = model_gb.predict(x_test)


# In[69]:


from sklearn.metrics import confusion_matrix
# 위쪽이 예측값, 좌측이 실제값
confusion_matrix(y_test, y_pred_gb)


# In[70]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_gb))

# 결과 : 정확도(0.93)

#%%

###############################################################################
# ### 2-7. 랜덤포레스트
###############################################################################

# > 랜덤 포레스트(Random Forest)는 배깅(Bagging)을 적용한 의사결정나무(Decision Tree)의 앙상블 알고리즘입니다. 
# 랜덤 포레스트는 나무(tree)가 아니라 나무가 모인 숲(forest)의 수준으로 하나의 트리 모델이 아닌 
# 다수의 부트스트랩 표본으로 트리 모델을 만든 후 그 결과를 취합해 분류(classification)의 경우에는 다수결로, 
# 회귀(regression)의 경우에는 평균을 출력합니다. 
# 이는 배깅과 동일한 방법이나 트리 모델의 분할 기준(노드)을 정하는 방법에서 차이가 있습니다.  
# 배깅은 노드(node)마다 모든 독립변수 내에서 최적의 분할을 찾는 방법을 사용하지만, 
# 랜덤 포레스트는 독립변수들을 무작위(random)로 추출하고, 
# 추출된 독립변수 내에서 최적의 분할을 만들어 나가는 방법을 사용합니다.  
# 일반적으로 하나의 트리 모델에서 발생할 수 있는 과적합(overfitting) 문제가 랜덤 포레스트에서는 줄어들고, 
# 예측 성능 또한 높게 나옵니다.

# In[71]:


# 랜덤포레스트 알고리즘 수행을 위한 함수 불러오기
from sklearn.ensemble import RandomForestClassifier


# In[72]:


# 모델 구축 및 학습
model_rf = RandomForestClassifier().fit(x_train, y_train)


# In[73]:


y_pred_rf = model_rf.predict(x_test)


# In[74]:


from sklearn.metrics import confusion_matrix
# 위쪽이 예측값, 좌측이 실제값
confusion_matrix(y_test, y_pred_rf)


# In[75]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_rf))

# 결과 : 정확도(0.93)

#%%

# ※ 예측결과를 확률로 출력하는 방법  
# sklearn 패키지의 분류 알고리즘을 이용해 모델을 만들어 학습(fit)시킨 후 
# 일반적으로 predict() 메소드를 이용해 예측결과를 클래스 형태로 출력하지만, 
# predict_proba() 메소드를 이용하면 확률 형태로도 출력할 수 있습니다.  
# (예시)  
y_prob_rf = model_rf.predict_proba(x_test)  
y_prob_rf[0:5]

#%%

###############################################################################
# ### 2-8. 서포트벡터머신
###############################################################################

# 서포트 벡터 머신(SVM, Support Vector Machine)은 고차원의 공간에서 
# 최적의 분리 초평면(hyperplane)을 찾아 이를 이용해 
# 분류(classification)와 회귀(regression)를 수행하는 알고리즘입니다.

# In[76]:


# 서포트 벡터머신 알고리즘 수행을 위한 함수 불러오기
from sklearn.svm import SVC


# In[77]:

# 커널함수:
# 데이터를 선형으로는 구분이 불가능 할 수 있다.
# 2차원의 데이터를 3차원으로 변환하면 결정경계를 찾을 수도 있다.
# 이런 방법을 커널 트릭(Kernel Trick)이라 한다.
# 비선형 분류를 위해 데이터를 더 높은 차원으로 변환시키는 함수를 커널 함수(Kernel Function)라 한다.
# 커널함수: 선형(linear), 다항(polynormial), 가우시안(Gaussian), RBF(Redial Basic Function), 시그모이드(Sigmoid) 등

# 기본 커널: 'rbf'
# 모델 구축(kernel:linear) 및 학습
model_svm = SVC(kernel = 'linear').fit(x_train, y_train)


# In[78]:


y_pred_svm = model_svm.predict(x_test)


# In[79]:


from sklearn.metrics import confusion_matrix
# 위쪽이 예측값, 좌측이 실제값
confusion_matrix(y_test, y_pred_svm)

#%%

"""
array([[20,  0,  0],
       [ 1, 17,  2],
       [ 0,  1, 19]], dtype=int64)
"""

# In[80]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_svm))

#%%

# 정확도: 0.93
# 정밀도: 0.90(최소)
# 민감도: 0.85(최소)

"""
              precision    recall  f1-score   support

           0       0.95      1.00      0.98        20
           1       0.94      0.85      0.89        20
           2       0.90      0.95      0.93        20

    accuracy                           0.93        60
   macro avg       0.93      0.93      0.93        60
weighted avg       0.93      0.93      0.93        60
"""

# In[81]:


# 예측값을 데이터프레임으로 만들고, 컬럼명을 breeds_pred로 지정
df_y_pred_svm = pd.DataFrame(y_pred_svm, columns = ['breeds_pred'])


# In[82]:


# 기존 test 데이터 셋에 svm 예측 결과 열합치기
df_svm = pd.concat([df_test, df_y_pred_svm], axis = 1)


# In[83]:


df_svm.head()


# In[84]:


# 실제값 및 예측값 시각화
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize = (10,5))
plt.subplot(1, 2, 1)
sns.scatterplot(data = df_svm, x='tail_length', y='wing_length', hue='breeds', style='breeds')
plt.title('actual')
plt.subplot(1, 2, 2)
sns.scatterplot(data = df_svm, x='tail_length', y='wing_length', hue='breeds_pred', style='breeds_pred')
plt.title('predicted')
plt.show()


#%%

###############################################################################
# ### 2-9. XGBoost
###############################################################################

# ① XGBoost

# In[85]:


# xgboost 패키지 설치(anaconda 기본 패키지가 아님)
# get_ipython().system('pip install xgboost')


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


# In[90]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_xgb))


# ② 그리드서치를 이용한 하이퍼 파라미터 튜닝

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


# In[96]:


# 정확도 최고일 때 하이퍼 파라미터
xgb_grid.best_params_


# In[97]:


# 선정된 하이퍼 파라미터로 재학습
model_xgb2 = XGBClassifier(eta=0.5, gamma =0, max_depth=8).fit(x_train, y_train)


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

# ## 3. 효과적인 사육을 위해 사육환경을 분리해보자 (군집 알고리즘)

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


import matplotlib.pyplot as plt
plt.figure(figsize = (10,7))
plt.scatter(x = cl['food'], y = cl['weight'])
plt.show()


# In[106]:


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


# In[111]:


plt.figure(figsize = (10,7))
plt.scatter(x = cl['food'], y = cl['weight'], c = cl_kmc.labels_)
plt.show()

#%%

# THE END