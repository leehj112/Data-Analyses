# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:23:11 2024

@author: leehj
"""

#%%
# 분류 알고리즘 - 결정트리(Decision Tree)
# 의사결정나무
# 관측값과 목표값을 연결시켜주는 예측모델
# 나무구조(tree), 분기점(node), 가지(branch)
# 선형모델과는 다른 특징을 가짐 - 선형은 기울기를 찾음
# 특정지점을 기준으로 분류를 하면서 그룹을 나눔
# 예측력이나 성능은 뛰어나지 않지면 시각화의 장점(설명력)
# 설명력: 중요한 요인을 밝히는데 장점(발병률)
#
# 장점: 데이터에 대한 가정이 없는 모델    
# 단점: 
#   - 트리가 무한정 깊어 지면 오버피팅 문제(과적합)    
#   - 예측력이 떨어짐
#
# sklearn.tree.DecisionTreeClassifier()
# sklearn 라이브러리에서 Decision Tree 분류 모형 가져오기

# 모형 객체 생성 (criterion='entropy' 적용)
# criterion: 'gini' 기본값, 'entropy' 
#   - 노드의 순도를 평가하는 방법
#   - 노드의 순도가 높을수록 지니나 엔트로피 값은 낮아 진다.
#   - 'gini' 불순도, 결정트리가 최적의 질문을 찾기 위한 기준
# max_depth: None, 기본값
#
# 평가:
#   - 과대적합(오버피팅): 모델이 훈련데이터에 지나치게 잘 맞도록 학습되어 예측력이 떨어짐
#   - 과소적합(언더피팅): 모델이 충분히 학습되지 않아 훈련데이터 대해서 예측력이 떨어짐


#%%
# 붓꽃의 품종 판별

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

#%%
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

# P95
# ### 의사결정나무(Decision Tree)
# 가지 : branch
# 분기점 : node
# 잎노드 : leaf node,  최종적으로 갖게되는 노드
# 가장 빈도수가 높은 클래스를 예측값으로 분류
#    -> 많은 잎노드를 갖는 것을 선택
# 깊이 : max_depth

# In[35]:


# 모델 학습 및 예측
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth=3, random_state=20)
dtc.fit(X_train, y_train)


# In[36]:


# 예측
y_dtc_pred = dtc.predict(X_test)
print("예측값: ", y_dtc_pred[:5])
# 성능 평가
dtc_acc = accuracy_score(y_test, y_dtc_pred)
print("Accuracy: %.4f" % dtc_acc) # 0