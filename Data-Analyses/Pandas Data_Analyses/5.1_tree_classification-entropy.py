# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:58:49 2024

@author: leehj
"""

# 결정트리(Decision Tree)
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
# 기본 라이브러리 불러오기
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import numpy as np

#%%
'''
[Step 1] 데이터 준비/ 기본 설정
'''

#%%
# https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original

#%%

# Breast Cancer 데이터셋 가져오기 (출처: UCI ML Repository)
uci_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
df = pd.read_csv(uci_path, header=None)

# 열 이름 지정
df.columns = ['id', 'clump', 'cell_size', 'cell_shape', 'adhesion', 'epithlial',
              'bare_nuclei', 'chromatin', 'normal_nucleoli', 'mitoses', 'class']

#  IPython 디스플레이 설정 - 출력할 열의 개수 한도 늘리기
pd.set_option('display.max_columns', 15)


#%%
'''
[Step 2] 데이터 탐색
'''

# 데이터 살펴보기
print(df.head())
print('\n')

#%%
# 데이터 자료형 확인
print(df.info())
print('\n')

#%%
# 데이터 통계 요약정보 확인
print(df.describe())
print('\n')

#%%

# bare_nuclei 열의 자료형 변경 (문자열 ->숫자)
# bare_nuclei 열의 고유값 확인
print(df['bare_nuclei'].unique())
print('\n')

# ['1' '10' '2' '4' '3' '9' '7' '?' '5' '8' '6']

#%%

# df['bare_nuclei'].replace('?', np.nan, inplace=True)      # '?'을 np.nan으로 변경
df['bare_nuclei'] = df['bare_nuclei'].replace({'?': np.nan})      # '?'을 np.nan으로 변경

#%%

# 누락데이터 행을 삭제
df.dropna(subset=['bare_nuclei'], axis=0, inplace=True)   

#%%

df['bare_nuclei'] = df['bare_nuclei'].astype('int')       # 문자열을 정수형으로 변환

print(df.describe())                                      # 데이터 통계 요약정보 확인
print('\n')

#%%
'''
[Step 3] 데이터셋 구분 - 훈련용(train data)/ 검증용(test data)
'''

x_cols = ['clump', 'cell_size', 'cell_shape', 
          'adhesion', 'epithlial', 'bare_nuclei', 
          'chromatin', 'normal_nucleoli', 'mitoses']

# 속성(변수) 선택
# 컬럼('id') 제외
X = df[x_cols] # 설명 변수 X
y = df['class']  # 예측 변수 Y

#%%

print(X.shape) # (683, 9)
print(y.shape) # (683,)

#%%

class_unique = df['class'].unique()
print("class:", class_unique) # [2 4]
# 2 : 양성
# 4 : 악성

#%%

# 설명 변수 데이터를 정규화
X = preprocessing.StandardScaler().fit(X).transform(X)

# train data 와 test data로 구분(7:3 비율)
# shuffle : True, 기본값, 데이터를 분할 할 때 섞음
# test_size : 0.3, 검증 데이터 비율(30%)
# random_state : 10, 데이터를 분할 할 때 섞는 랜덤 시드값
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

print('훈련 데이터 개수: ', X_train.shape) # (478, 9)
print('검증 데이터 개수: ', X_test.shape)  # (205, 9)
print('\n')

#%%
'''
[Step 4] Decision Tree 분류 모형 - sklearn 사용
'''

# sklearn 라이브러리에서 Decision Tree 분류 모형 가져오기

# 모형 객체 생성 (criterion='entropy' 적용)
# criterion: 'gini' 기본값, 'entropy' 
#   - 노드의 순도를 평가하는 방법
#   - 노드의 순도가 높을수록 지니나 엔트로피 값은 낮아 진다.
#   - 'gini' 불순도, 결정트리가 최적의 질문을 찾기 위한 기준
# max_depth: None, 기본값
tree_model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)

#%%

# train data를 가지고 모형 학습
tree_model.fit(X_train, y_train)

#%%
# test data를 가지고 y_hat을 예측 (분류)
y_hat = tree_model.predict(X_test)      # 2: benign(양성), 4: malignant(악성)

#%%
print(y_hat[0:10])
print(y_test.values[0:10])
print('\n')

#%%

# 모형 성능 평가 - Confusion Matrix 계산
tree_matrix = metrics.confusion_matrix(y_test, y_hat)
print(tree_matrix)
print('\n')

#%%
"""
[[125   7]
 [ 14  59]]
"""

#%%
"""
# Confusion Matrix
# 정확도(precision) = TP / (TP + FP)
# 재현율(recall) = TP / (TP + FN)
# F1 지표 = (2 * (recision * recall)) / (precision + recall)
###################################################
               예측값
               False   True
-------------+-------------------------       
실제값 False | TN      FP
       True  | FN      TP

###################################################
TN(True Negative):  음성을 음성으로 판단, 정확함
TP(True Positive):  양성을 양성으로 판단, 정확함
FP(False Positive): 음성을 양성으로 판단, 1종 오류
FN(False Negaive):  양성을 음성으로 판단, 2종 오류
###################################################
"""

#%%
# 모형 성능 평가 - 평가지표 계산
tree_report = metrics.classification_report(y_test, y_hat)
print(tree_report)

#%%

"""
              precision    recall  f1-score   support

           2       0.98      0.97      0.98       131
           4       0.95      0.97      0.96        74

    accuracy                           0.97       205
   macro avg       0.97      0.97      0.97       205
weighted avg       0.97      0.97      0.97       205
"""

#%%

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(30,15))
plot_tree(tree_model) # 출력할 모델 지정
plt.show()

#%%

plt.figure(figsize=(40,15))
plot_tree(tree_model, fontsize=25)
plt.show()

#%%

plt.figure(figsize=(40,15))
plot_tree(tree_model, fontsize=20, feature_names=x_cols, filled=True)
plt.show()