# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:20:19 2024

@author: leehj
"""

# 붓꽃의 품종 판별
# pip install xgboost

# 데이터셋 불러오기
# 다중분류
# 0: Setosa
# 1: Versicolor
# 2: Virginica

# Dataset
# sepal length, sepal width, petal length, petal width

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

# SVM(Support Vector Machine)
# 벡터 공간을 나누어 경계를 찾음
# 목표 클래스별 군집

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
print("데이터셋 타입:", type(iris['data'])) # <class 'numpy.ndarray'>
print("데이터셋 크기:", iris['data'].shape) # (150, 4)
print("\t 행의 갯수:", iris['data'].shape[0])
print("\t 열의 갯수::", iris['data'].shape[1])

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
df.head(2) # 2개의 행을 출력


# In[8]:


# Target 열 추가
df['Target'] = iris['target']
print('데이터셋의 크기: ', df.shape)
df.head() # 5개의 행을 출력


# # 데이터 탐색(EDA)

# In[9]:


# 데이터프레임의 기본정보
df.info()


# In[10]:


# 통계정보 요약
df.describe()


# In[11]:


# 결측값 확인
# 칼럼별로 집계
df.isnull().sum()


# In[12]:


# 중복 데이터 확인
# 행별 중복 유무: 중복(True)
print(df.duplicated()) # 결과: True, False

df.duplicated().sum()


# In[13]:


# 중복 데이터 출력
df.loc[df.duplicated(), :]


# In[14]:


# 중복 데이터 모두 출력
df.loc[(df.sepal_length==5.8)&(df.petal_width==1.9), :]
# 중복행: 101, 142

#%%

# 정렬
dfs = df.sort_values(['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'Target'])
dfs[df.duplicated(keep=False)] # 중복된 행 모두 출력

#%%

#
df_sepal_length = df['sepal_length'].sort_values()
df_sepal_width = df['sepal_width'].sort_values()
df_petal_length = df['petal_length'].sort_values()
df_petal_width = df['petal_width'].sort_values()
df_Target = df['Target'].sort_values()

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

"""
Target
0    50
1    50
2    49
Name: count, dtype: int64
"""

# In[20]:

# sepal_length 값의 분포 - hist 함수
plt.hist(x='sepal_length', data=df)
plt.show()


# In[21]:


# sepal_width 값의 분포 - displot 함수 (histogram)
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


###############################################################################
# # Baseline 모델 학습

# #### 학습용-테스트 데이터셋 분리하기

# In[26]:


from sklearn.model_selection import train_test_split

X_data = df.loc[:, 'sepal_length':'petal_width'] # 훈련 데이터
y_data = df.loc[:, 'Target']    # 정답

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, 
                                                    test_size=0.2, # 테스트 데이터 20%
                                                    shuffle=True,  # 데이터를 섞음
                                                    random_state=20) # 난수

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


#%%
#######################################################################
# ### SVM(Support Vector Machine)
# 백터 공간을 나누는 경계를 찾음

from sklearn.metrics import accuracy_score

# In[30]:


# 모델 학습
from sklearn.svm import SVC

# RBF(Radial Basic Function) 가우시안 함수에 기반한 비선형 대응
svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)

# In[31]:


# 예측
y_svc_pred = svc.predict(X_test)
print("예측값: ", y_svc_pred[:5])
# 성능 평가
svc_acc = accuracy_score(y_test, y_svc_pred)
print("Accuracy: %.4f" % svc_acc)

#%%

# 예측 확인
print("예측값: ", y_svc_pred[:])
print("정답값: ", y_test[:].values)

#%%

# 정답이 틀린위치
yn = y_svc_pred != y_test
print(yn) # 모두 False이므로 모두 정답 

#%%

print("틀린 위치 : ", yn[yn == True])
print("예측값: ", y_svc_pred[yn])    # []
print("정답값: ", y_test[yn].values) # []