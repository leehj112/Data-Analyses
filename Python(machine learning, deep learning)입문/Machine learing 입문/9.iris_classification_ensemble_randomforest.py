# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 09:09:33 2024

@author: leehj
"""

# 앙상블 모델 - 배깅(bagging)
# 랜덤 포레스트(Random Forest)는 결정트리(DecisionTree) 모델을 여러 개 사용하여 각 모델의 예측값을 보팅(투표)하여 결정
# 같은 종류의 알고리즘 모델을 여러 개 결합하여 예측하는 방법을 배깅(baggin)이라 한다.
# 각 트리는 전체 학습 데이터 중에서 서로 다른 데이터를 샘플링하여 학습
#
# 앙상블 기법 : 여러 모델을 만들어 각 예측값들을 투표 또는 평균 등으로 통합하여 예측
# 작동방식:
#    - 결정트리(DecisionTree) 모델을 여러 개 생성
#    - 전체 학습 데이터 중에서 서로 다른 데이터를 샘플링
#    - 각 모델의 예측값들의 평균값을 구해서 예측

# 랜덤 포레스트의 특징:
#    - 결정트리의 단점으로 트리가 무한정 깊어 지면 오버피팅 문제(과적합) 발생
#    - 오버피팅 문제를 완화
#    - 랜덤하게 생성된 많은 트리를 이용하여 예측
#    - 결정트리와 마찬가지로 아웃라이어에 거의 영향을 받지 않는다.
#    - 선형이나 비선형 데이터에 상관없이 잘 동작
#
# 랜덤 포레스트의 단점:
#    - 학습 속도가 상대적으로 느리다.
#    - 수많은 트리를 동원하기 때문에 모델에 대한 해석이 어렵다.    
#
# 유용성:
#    - 종속변수(y)가 연속형 데이터와 범주형 데이터인 경우 모두 사용할 수 있다.(선형, 비선형)
#    - 아웃라이어가 문제가 되는 경우 선형 모델보다 좋은 대안이 될 수 있다.
#    - 오버피팅 문제로 결정 트리를 사용하기 어려울 때 대안이 될 수 있다.
# 
# 
# 모듈: from sklearn.ensemble import RandomForestClassifier
# 예시: RandomForestClassifier(n_estimators=50, max_depth=3, random_state=20)

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

from sklearn.metrics import accuracy_score

#%%

# ### 배깅(Bagging): (랜덤포레스트)
# Decision Tree 모델을 여러 개 결합
# 서로 다른 데이터를 샘플링
# n_estimators: 트리 모델의 갯수
# max_depth: 개별 드리의 깊이

#%%

# 모델 학습 및 검증
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=20)

# 훈련
rfc.fit(X_train, y_train)

#%%
# 예측
y_rfc_pred = rfc.predict(X_test)
print("예측값: ", y_rfc_pred[:5])

# 모델 성능 평가
rfc_acc = accuracy_score(y_test, y_rfc_pred)
print("Accuracy: %.4f" % rfc_acc) # 0.9667

#%%

# 결과:
#   - 단일 결정트리:  0.9333
#   - 랜덤 포레스트:  0.9667 (향상)

#%%