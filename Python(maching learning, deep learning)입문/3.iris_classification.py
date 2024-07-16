# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:19:03 2024

@author: leehj
"""

# 라이브러리 환경
import pandas as pd
import numpy as np


# In[2]:


# skleran 데이터셋에서 iris 데이터셋 로딩
from sklearn import datasets
iris = datasets.load_iris()

# iris 데이터셋은 딕셔너리 형태이므로, key 값을 확인
iris.keys()
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

# In[3]:


# DESCR 키를 이용하여 데이터셋 설명(Description) 출력
print(iris['DESCR'])

#%%
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

# In[4]:


# target 속성의 데이터셋 크기
print("데이터셋 크기:", iris['target'].shape) # 0 1 2

# target 속성의 데이터셋 내용
print("데이터셋 내용: \n", iris['target'])


# In[5]:

# data 속성의 데이터셋 크기
print("데이터셋 크기:", iris['data'].shape) # (150, 4)

#%%

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

#%%
"""
     sepal_length  sepal_width  petal_length  petal_width  Target
142           5.8          2.7           5.1          1.9       2
"""

#%%

# 중복 데이터를 모두 추출해서 새로운 데이터프레임
df_dup = df.loc[df.duplicated(keep=False), :]
print(df_dup)

#%%

"""
     sepal_length  sepal_width  petal_length  petal_width  Target
101           5.8          2.7           5.1          1.9       2
142           5.8          2.7           5.1          1.9       2
"""

# In[14]:

# 중복 데이터 모두 출력
df.loc[(df.sepal_length==5.8)&(df.petal_width==1.9), :]

#%%
"""
     sepal_length  sepal_width  petal_length  petal_width  Target
101           5.8          2.7           5.1          1.9       2
142           5.8          2.7           5.1          1.9       2
"""

# In[15]:


# 중복 데이터 제거
df = df.drop_duplicates()
df.loc[(df.sepal_length==5.8)&(df.petal_width==1.9), :]

#%%

print(df_dup.index)
# Index([101, 142], dtype='int64')

#%%

# 중복된 데이터에서 삭제 처리된 행 출력
for n in df_dup.index:
    try:
        print(n, df.loc[n,:])
    except KeyError as e:
        print("## 삭제된 행:", e)
        

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

#%%
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
# # Baseline 모델 학습

# #### 학습용-테스트 데이터셋 분리하기

# In[26]:


from sklearn.model_selection import train_test_split

X_data = df.loc[:, 'sepal_length':'petal_width']
y_data = df.loc[:, 'Target']

# 검증용 : 20%
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, 
                                                    test_size=0.2, 
                                                    shuffle=True, 
                                                    random_state=20)

print("훈련 데이터:", X_train.shape, y_train.shape) # (119, 4) (119,)
print("검증 데이터:", X_test.shape, y_test.shape)   # (30, 4) (30,)


# ### KNN

# In[27]:

# 모델 학습 : kNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)


# In[28]:


# 예측
y_knn_pred = knn.predict(X_test)
print("예측값: ", y_knn_pred[:])
print("정답값: ", y_test[:].values)

#%%

# 정답이 틀린위치
yn = y_knn_pred != y_test
print(yn) # 72

#%%

print("틀린 위치 : ", yn[yn == True])
print("예측값: ", y_knn_pred[yn])
print("정답값: ", y_test[yn].values)


# In[29]:


# 성능 평가
from sklearn.metrics import accuracy_score
knn_acc = accuracy_score(y_test, y_knn_pred)
print("Accuracy: %.4f" % knn_acc) # Accuracy: 0.9667


# ### SVM

# In[30]:


# 모델 학습
from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)


# In[31]:


# 예측
y_svc_pred = svc.predict(X_test)
print("예측값: ", y_svc_pred[:5])
# 성능 평가
svc_acc = accuracy_score(y_test, y_svc_pred)
print("Accuracy: %.4f" % svc_acc)


# ### 로지스틱 회귀

# In[32]:


# 모델 학습
from sklearn.linear_model import LogisticRegression
lrc = LogisticRegression()
lrc.fit(X_train, y_train)


# In[33]:


# 예측
y_lrc_pred = lrc.predict(X_test)
print("예측값: ", y_lrc_pred[:5])
# 성능 평가
lrc_acc = accuracy_score(y_test, y_lrc_pred)
print("Accuracy: %.4f" % lrc_acc)


# In[34]:


# 확률값 예측
y_lrc_prob = lrc.predict_proba(X_test)
y_lrc_prob


# ### 의사결정나무

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
print("Accuracy: %.4f" % dtc_acc)


# #앙상블 모델

# ### 보팅

# In[37]:


# Hard Voting 모델 학습 및 예측
# voting='hard' : 다수결로 최종 분류 결정
from sklearn.ensemble import VotingClassifier
hvc = VotingClassifier(estimators=[('KNN', knn), ('SVM', svc), ('DT', dtc)], voting='hard')
hvc.fit(X_train, y_train)
# 예측
y_hvc_pred = hvc.predict(X_test)
print("예측값: ", y_hvc_pred[:5])
# 성능 평가
hvc_acc = accuracy_score(y_test, y_hvc_pred)
print("Accuracy: %.4f" % hvc_acc)

#%%

# ### 배깅 (랜덤포레스트)

# In[38]:


# 모델 학습 및 검증
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=20)
rfc.fit(X_train, y_train)
# 예측
y_rfc_pred = rfc.predict(X_test)
print("예측값: ", y_rfc_pred[:5])
# 모델 성능 평가
rfc_acc = accuracy_score(y_test, y_rfc_pred)
print("Accuracy: %.4f" % rfc_acc)


# ### 부스팅 (XGBoost)

# In[39]:


# 모델 학습 및 예측
from xgboost import XGBClassifier
xgbc = XGBClassifier(n_estimators=50, max_depth=3, random_state=20)
xgbc.fit(X_train, y_train)
# 예측
y_xgbc_pred = xgbc.predict(X_test)
print("예측값: ", y_xgbc_pred[:5])
# 모델 성능 평가
xgbc_acc = accuracy_score(y_test, y_xgbc_pred)
print("Accuracy: %.4f" % xgbc_acc)


# # 교차 검증 (Cross-Validation)

# ### Hold out 교차 검증

# In[40]:


# 검증용 데이터셋 분리
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, 
                                            test_size=0.3, 
                                            shuffle=True, 
                                            random_state=20)
print(X_tr.shape, y_tr.shape)
print(X_val.shape, y_val.shape)


# In[41]:


# 학습
rfc = RandomForestClassifier(max_depth=3, random_state=20)
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
print("Test Accuracy: %.4f" % test_acc)


# ### K-Fold 교차 검증

# In[43]:


# 데이터셋을 5개의 Fold로 분할하는 KFold 클래스 객체 생성
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=20)
# 훈련용 데이터와 검증용 데이터의 행 인덱스를 각 Fold별로 구분하여 생성
num_fold = 1
for tr_idx, val_idx in kfold.split(X_train): 
    print("%s Fold----------------------------------" % num_fold)
    print("훈련: ", len(tr_idx), tr_idx[:10])
    print("검증: ", len(val_idx), val_idx[:10])
    num_fold = num_fold + 1


# In[44]:


# 훈련용 데이터와 검증용 데이터의 행 인덱스를 각 Fold별로 구분하여 생성
val_scores = []
num_fold = 1
for tr_idx, val_idx in kfold.split(X_train, y_train):
    # 훈련용 데이터와 검증용 데이터를 행 인덱스 기준으로 추출
    X_tr, X_val = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
    # 학습
    rfc = RandomForestClassifier(max_depth=5, random_state=20)
    rfc.fit(X_tr, y_tr)
    # 검증
    y_val_pred = rfc.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)  
    print("%d Fold Accuracy: %.4f" % (num_fold, val_acc))
    val_scores.append(val_acc)   
    num_fold += 1  


# In[45]:


# 평균 Accuracy 계산
import numpy as np
mean_score = np.mean(val_scores)
print("평균 검증 Accuraccy: ", np.round(mean_score, 4))