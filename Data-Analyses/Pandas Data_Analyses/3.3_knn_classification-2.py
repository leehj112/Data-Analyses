# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:43:25 2024

@author: leehj
"""

# 지도학습
# 분류(classfication)
# 목표변수가 갖고 있는 카테고리(범주형) 값 중에서 분류 예측
# 고객분류, 질병 진단, 스펨 메일 필터링, 음성인식
# 알고리즘: KNN, SVM, Decision Tree, Logistic Regression

# KNN(k-Nearest Neighbors)
# k개의 가까운 이웃

#%%

# Confusion Matrix
# 정확도(precision) = TP / (TP + FP)
# 재현율(recall) = TP / (TP + FN)
# F1 지표 = (2 * (recision * recall)) / (precision + recall)
"""
               예측값
               False   True
-------------+-------------------------       
실제값 False | TN      FP
       True  | FN      TP

"""
"""
TN(True Negative):  음성을 음성으로 판단, 정확함
TP(True Positive):  양성을 양성으로 판단, 정확함
FP(False Positive): 음성을 양성으로 판단, 1종 오류
FN(False Negaive):  양성을 음성으로 판단, 2종 오류
"""

#%%
# 타이타닉 승객들의 생존여무(survived) 예측

### 기본 라이브러리 불러오기
import pandas as pd
import seaborn as sns

'''
[Step 1] 데이터 준비 - Seaborn에서 제공하는 titanic 데이터셋 가져오기
'''

# load_dataset 함수를 사용하여 데이터프레임으로 변환
df = sns.load_dataset('titanic')

# 데이터 살펴보기
print(df.head())   
print('\n')

#  IPython 디스플레이 설정 - 출력할 열의 개수 한도 늘리기
pd.set_option('display.max_columns', 15)
print(df.head())   
print('\n')

#%%
'''
[Step 2] 데이터 탐색
'''

# 데이터 자료형 확인
print(df.info())  
print('\n')

#%%

# NaN값이 많은 deck 열을 삭제, 
# embarked와 내용이 겹치는 embark_town 열을 삭제
rdf = df.drop(['deck', 'embark_town'], axis=1)  
print(rdf.columns.values)
print('\n')

#%%

# age 열에 나이 데이터가 없는 모든 행을 삭제 - age 열(891개 중 177개의 NaN 값)
rdf = rdf.dropna(subset=['age'], how='any', axis=0)  
print(len(rdf)) # 714 = 891 - 177
print('\n')

#%%

# embarked 열의 NaN값을 승선도시 중에서 가장 많이 출현한 값으로 치환하기
most_freq = rdf['embarked'].value_counts(dropna=True).idxmax()   
print(most_freq) # 'S'
print('\n')

#%%

print(rdf.describe(include='all'))
print('\n')

#%%

# 컬럼('embarked')의 누락데이터는 가장 많은 빈도수를 가진 'S'로 채움
# rdf['embarked'].fillna(most_freq, inplace=True)
rdf['embarked'] = rdf['embarked'].fillna(most_freq)

#%%

rdf['embarked'].value_counts()

#%%

"""
embarked
S    556
C    130
Q     28
Name: count, dtype: int64
"""

#%%

'''
[Step 3] 분석에 사용할 속성을 선택
'''

# 분석에 활용할 열(속성)을 선택 
ndf = rdf[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'embarked']]
print(ndf.head())   
print('\n')

#%%

# 원핫인코딩 - 범주형 데이터를 모형이 인식할 수 있도록 숫자형으로 변환
onehot_sex = pd.get_dummies(ndf['sex'])

#%%
ndf = pd.concat([ndf, onehot_sex], axis=1)

#%%

onehot_embarked = pd.get_dummies(ndf['embarked'], prefix='town')
ndf = pd.concat([ndf, onehot_embarked], axis=1)

#%%
ndf.drop(['sex', 'embarked'], axis=1, inplace=True)
print(ndf.head())   
print('\n')

#%%
'''
[Step 4] 데이터셋 구분 - 훈련용(train data)/ 검증용(test data)
'''

# 속성(변수) 선택
# 독립 변수 X
X=ndf[['pclass', 'age', 'sibsp', 'parch', 'female', 'male', 'town_C', 'town_Q', 'town_S']]  

# 종속 변수 Y
y=ndf['survived']                      

#%%

# 설명 변수 데이터를 정규화(normalization)
# 표준정규분포 = (x - 평균(x)) / 표준편차(x)
from sklearn import preprocessing

X0 = preprocessing.StandardScaler().fit(X)
X = preprocessing.StandardScaler().fit(X).transform(X)

#%%
# train data 와 test data로 구분(7:3 비율)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10) 

print('훈련 데이터 개수: ', X_train.shape) # (499, 9)
print('검증 데이터 개수: ', X_test.shape)  # (215, 9)

#%%

###############################################################################
# [Step 5] KNN 분류 모형 - sklearn 사용
###############################################################################

#%%

#%%
# sklearn 라이브러리에서 KNN 분류 모형 가져오기
from sklearn.neighbors import KNeighborsClassifier

# n_neighbors: 5, 기본값
# 모형 객체 생성 (k=5로 설정)
knn = KNeighborsClassifier(n_neighbors=5)

# train data를 가지고 모형 학습
knn.fit(X_train, y_train)   

# test data를 가지고 y_hat을 예측 (분류) 
y_hat = knn.predict(X_test)

print('예측:', y_hat[0:20])
print('정답:', y_test.values[0:20])

#%%

# 결정계수(R-제곱)
# 학습을 마친 모형에 검증 데이터를 적용하여 결정계수(R-제곱) 계산
r_square = knn.score(X_test, y_test)
print(r_square) # 0.8093023255813954
print('\n')

#%%

# help(KNeighborsClassifier)
help(knn)

#%%

###############################################################################
# 모형 성능 평가 - Confusion Matrix 계산
###############################################################################

from sklearn import metrics 

# 검증용 정답과 예측 정답
knn_matrix = metrics.confusion_matrix(y_test, y_hat)  
print(knn_matrix)

#%%

# Confusion Matrix
"""
[[110  15]
 [ 26  64]]
"""

"""
               예측값
               False    True
-------------+-------------------------       
실제값 False | TN      FP
       True  | FN      TP

"""

#%%

# 모형 성능 평가 - 평가지표 계산
knn_report = metrics.classification_report(y_test, y_hat)            
print(knn_report)
#%%

"""
              정확도       재현율  F1 지표
              precision    recall  f1-score   support

           0       0.81      0.88      0.84       125 # 미생존자
           1       0.81      0.71      0.76        90 # 생존자

    accuracy                           0.81       215
   macro avg       0.81      0.80      0.80       215
weighted avg       0.81      0.81      0.81       215
"""
"""
[[110  15]
 [ 26  64]]
"""

#%%

# 교재 : P316
TP = knn_matrix[1,1]
FP = knn_matrix[0,1]
FN = knn_matrix[1,0]
TN = knn_matrix[0,0]

# 정확도(precision) = TP / (TP + FP)
precision = TP / (TP + FP)
print("precision: ", round(precision,2))

# 재현율(recall) = TP / (TP + FN)
recall = TP / (TP + FN)
recall = round(recall, 2)
print("recall: ", recall)

# F1 지표 = (2 * (recision * recall)) / (precision + recall)
F1_score = (2 * (precision * recall)) / (precision + recall)
F1_score = round(F1_score, 2)
print("F1_score:", F1_score)

#%%