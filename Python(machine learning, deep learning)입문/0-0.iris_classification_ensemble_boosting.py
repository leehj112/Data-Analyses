# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 09:10:29 2024

@author: leehj
"""

# 앙상블 모델 - 부스팅(boosting)
# XGBoost(eXtra Gradient Boost)
# 순차 점진적 향상 모델
# 부스팅 모델은 순차적으로 트리를 만들어 이전 트리로부터 더 나은 트리를 만들어 내는 알고리즘
# 트리를 순차적으로 만들면서 이전 트리에서 학습한 내용을 다음 트리를 만들 때 반영
# 랜덤 포레스트보다 빠르고 성능이 우수하다.
# 종류: XGBoost, LightGBM, CatBoost
#
# 장점: 
#   - 예측 속도가 빠르다.
#   - 예측력이 좋다.
#   - 변수 종류가 많고 데이터가 클수록 상대적으로 좋은 성능을 낸다.   
# 단점:    
#   - 해석이 어렵다.
#   - 하이퍼 파라미터 튜닝이 어렵다.
#
# 활용:
#   - 종족변수가 연속형 데이터나 범주형 모두 사용 가능
#   - 표 형태로 정리된 데이터인 경우 활용(이미지나 자연어가 제외)
#
# 모듈: from xgboost import XGBClassifier
# 예시: XGBClassifier(n_estimators=50, max_depth=5, random_state=20)

# 설명:
# 여러개의 약한 학습기를 순차적으로 학습
# 잘못 예측한 데이터에 대한 예측 오차를 줄일 수 있는 방향으로 모델을 계속 업데이트
# 캐글, 데이콘 등 경진대회에서 가장 많이 사용
# 모델 학습 속도가 빠르고 예측력이 높다.


#%%

# pip install scikit-learn===1.1.3
# scikit-learn(1.3) 지원하지 않음

# xgboost-2.0.3
# pip install xgboost

#%%
# # 데이터셋 불러오기
"""
    - sepal length in cm # 꽃받침(길이)
    - sepal width in cm  # 꽃받침(너비)
    - petal length in cm # 꽃잎(길이)
    - petal width in cm  # 꽃잎(너비)
    - class:             # 종류(타겟)
            - Iris-Setosa
            - Iris-Versicolour
            - Iris-Virginica
"""
# In[1]:


# 라이브러리 환경
import pandas as pd
import numpy as np
import sklearn as sk

print(pd.__version__) # 2.2.1
print(np.__version__) # 1.26.4
print(sk.__version__) # 1.4.1.post1

# In[2]:


# skleran 데이터셋에서 iris 데이터셋 로딩
from sklearn import datasets
iris = datasets.load_iris()

# iris 데이터셋은 딕셔너리 형태이므로, key 값을 확인
iris.keys()


# In[3]:


# DESCR 키를 이용하여 데이터셋 설명(Description) 출력
print(iris['DESCR'])


# In[4]:


# target 속성의 데이터셋 크기
print("데이터셋 크기:", iris['target'].shape)

# target 속성의 데이터셋 내용
print("데이터셋 내용: \n", iris['target'])


# In[5]:


# data 속성의 데이터셋 크기
print("데이터셋 크기:", iris['data'].shape)
# data 속성의 데이터셋 내용 (첫 7개 행을 추출)
print("데이터셋 내용: \n", iris['data'][:7, :])


# In[6]:


# data 속성을 판다스 데이터프레임으로 변환
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
print("데이터프레임의 형태:", df.shape)
df.head()


# In[7]:


# 열(column) 이름을 간결하게 변경
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df.head(2)


# In[8]:


# Target 열 추가
df['Target'] = iris['target']
print('데이터셋의 크기: ', df.shape)
df.head()


# # 데이터 탐색(EDA)

# In[9]:


# 데이터프레임의 기본정보
df.info()


# In[10]:


# 통계정보 요약
df.describe()


# In[11]:


# 결측값 확인
df.isnull().sum()


# In[12]:


# 중복 데이터 확인
df.duplicated().sum()


# In[13]:


# 중복 데이터 출력
df.loc[df.duplicated(), :]


# In[14]:


# 중복 데이터 모두 출력
df.loc[(df.sepal_length==5.8)&(df.petal_width==1.9), :]


# In[15]:


# 중복 데이터 제거
df = df.drop_duplicates()
df.loc[(df.sepal_length==5.8)&(df.petal_width==1.9), :]


# In[16]:


# 변수 간의 상관관계 분석
df.corr()


# In[17]:


# 시각화 라이브러리 설정
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)


# In[18]:


# 상관계수 히트맵
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()


# In[19]:


# Target 값의 분포 - value_counts 함수
df['Target'].value_counts()


# In[20]:


# sepal_length 값의 분포 - hist 함수
plt.hist(x='sepal_length', data=df)
plt.show()


# In[21]:


# sepal_widgth 값의 분포 - displot 함수 (histogram)
sns.displot(x='sepal_width', kind='hist', data=df)
plt.show()


# In[22]:


# petal_length 값의 분포 - displot 함수 (kde 밀도 함수 그래프)
sns.displot(x='petal_width', kind='kde', data=df)
plt.show()


# In[23]:


# 품종별 sepal_length 값의 분포 비교
sns.displot( x='sepal_length', hue='Target', kind='kde', data=df)
plt.show()


# In[24]:


# 나머지 3개 피처 데이터를 한번에 그래프로 출력
for col in ['sepal_width', 'petal_length', 'petal_width']:
    sns.displot(x=col, hue='Target', kind='kde', data=df)
plt.show()


# In[25]:


# 두 변수 간의 관계
sns.pairplot(df, hue = 'Target', height = 2.5, diag_kind = 'kde')
plt.show()


#%%

###############################################################################
# # Baseline 모델 학습

# #### 학습용-테스트 데이터셋 분리하기

# In[26]:


from sklearn.model_selection import train_test_split

X_data = df.loc[:, 'sepal_length':'petal_width']
y_data = df.loc[:, 'Target']

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, 
                                                    test_size=0.2, 
                                                    shuffle=True, 
                                                    random_state=20)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


#%%


#%%

# ### 부스팅 (XGBoost)
# 여러개의 약한 학습기를 순차적으로 학습
# 잘못 예측한 데이터에 대한 예측 오차를 줄일 수 있는 방향으로 모델을 계속 업데이트
# 캐글, 데이콘 등 경진대회에서 가장 많이 사용
# 모델 학습 속도가 빠르고 예측력이 높다.

# In[39]:


# 모델 학습 및 예측
from xgboost import XGBClassifier
xgbc = XGBClassifier(n_estimators=50, max_depth=3, random_state=20)
xgbc.fit(X_train, y_train)

#%%
# 예측
y_xgbc_pred = xgbc.predict(X_test)
print("예측값: ", y_xgbc_pred[:5])

#%%

# 모델 성능 평가
from sklearn.metrics import accuracy_score
xgbc_acc = accuracy_score(y_test, y_xgbc_pred)
print("Accuracy: %.4f" % xgbc_acc) # 0.933