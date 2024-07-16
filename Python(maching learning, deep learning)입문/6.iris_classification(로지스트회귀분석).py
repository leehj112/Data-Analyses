# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:20:41 2024

@author: leehj
"""

# 분류알고리즘 - 로지스틱 회귀(LogisticRegression)
# 이진분류(Binary Classification)
# 다중분류(Multi-Calss Classification)
# 선형회귀 분석에 근간을 두고 있는 분류 알고리즘
# 장점:
#   - 선형회귀처럼 쉽다.
#   - 계수(기울기)를 사용해서 각 변수의 중요성을 파악    
# 단점:    
#   - 선형회귀분석을 근간으로 하기 때문에 선형관계가 아닌 데이터에 대해서는 예측력이 떨어진다.    

# 그래프 해석
# 1. 테스트조건
# 2. 불순도나 엔트로피
# 3. 총 샘플 수(samples)
# 4. 클래스별 샘플 수(value)
# 가지(branch) : 조건
# 1. 왼쪽: 만족(yes)
# 2. 오른쪽: 불만족(no)
# filled(True) 색상
# 1. 색상의 농도
# 2. 색상의 구분(클래스별 분류)
# 노드(node) : max_depth=5
# 1. 최상위 : 0번 노드
# 2. 하위노드: 1~5 노드

#%%
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
# sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True, cmap="coolwarm")
plt.show()


# In[19]:


# Target 값의 분포 - value_counts 함수
df['Target'].value_counts()


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

from sklearn.metrics import accuracy_score

# ### 로지스틱 회귀(Logistic Regression)
# 시그모이드 함수의 출력값 : 0~1 사이의 값
# 1에 가까우면 해당 클래스로 분류
# 0에 가까우면 분류에서 제외

# In[32]:


# 모델 학습
from sklearn.linear_model import LogisticRegression
lrc = LogisticRegression()
lrc.fit(X_train, y_train)


# In[33]:


# 예측
y_lrc_pred = lrc.predict(X_test)
print("예측값: ", y_lrc_pred[:5])

#%%

# 성능 평가
lrc_acc = accuracy_score(y_test, y_lrc_pred)
print("Accuracy: %.4f" % lrc_acc) # Accuracy: 1.0000


# In[34]:


# 모든 클래스(정답)에 대한 확률값
# 확률값 예측 : Probability estimates.
y_lrc_prob = lrc.predict_proba(X_test)
y_lrc_prob

#%%

for n in range(len(y_test)):
    print("정답(%d), 인덱스(%3d) :" %  (y_test.iloc[n], y_test.index[n]), end='')
    for p in y_lrc_prob[n]:
        print("[%.4f]" % (p), end='')
    print()