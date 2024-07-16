# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 09:11:02 2024

@author: leehj
"""

# 교차검증 - Hold-out(홀드아웃)
# 데이터 일부를 검증 데이터로 사용하는 방법
# 검증 데이터는 모델 학습에 사용하지 않은 데이터로 모델의 일반화 성능평가에 사용
# 테스트 데이터에 대한 예측력을 높임

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

print(X_train.shape, y_train.shape) # (119, 4) (119,)
print(X_test.shape, y_test.shape)   # (30, 4) (30,)


#%%

from sklearn.metrics import accuracy_score

#%%

# In[39]:

# # 교차 검증 (Cross-Validation)

# ### Hold out 교차 검증

# In[40]:

"""
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, 
                                                    test_size=0.2, 
                                                    shuffle=True, 
                                                    random_state=20)
"""    

# 검증용 데이터셋 분리(7:3)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, 
                                            test_size=0.3, 
                                            shuffle=True, 
                                            random_state=20)
print(X_tr.shape, y_tr.shape)   # (83, 4) (83,)
print(X_val.shape, y_val.shape) # (36, 4) (36,)


# In[41]:


# 학습
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(max_depth=3, random_state=20)
# rfc.fit(X_train, y_train)
rfc.fit(X_tr, y_tr)

# 예측
y_tr_pred = rfc.predict(X_tr)
y_val_pred = rfc.predict(X_val)

# 검증
tr_acc = accuracy_score(y_tr, y_tr_pred)
val_acc = accuracy_score(y_val, y_val_pred)
print("Train Accuracy: %.4f" % tr_acc)
print("Validation Accuracy: %.4f" % val_acc)


# In[42]:


# 테스트 데이터 예측 및 평가
y_test_pred = rfc.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print("Test Accuracy: %.4f" % test_acc