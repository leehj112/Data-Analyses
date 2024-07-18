# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:36:27 2024

@author: leehj
"""

# -*- coding: utf-8 -*-
"""
[k-최근접 이웃 분류]
pip install mglearn

target_name : 0(malignant, 악성), 1(benign, 양성)

"""
# 유방암 데이터

from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_breast_cancer
import mglearn
import matplotlib.pyplot as plt

cancer = load_breast_cancer()

# 훈련데이터와 테스트 데이터를 분할하는 방법
# 한쪽으로 데이터가 편중되는 것을 방지
# 정답 데이터를 기준으로 분할
# stratify : cancer.target
X_train, X_test, y_train, y_test = train_test_split(
   cancer.data, cancer.target, stratify=cancer.target, random_state=66)

#%%
training_accuracy = []
test_accuracy = []

# 1에서 10까지 n_neighbors를 적용
neighbors_settings = range(1, 11) 

for n_neighbors in neighbors_settings:
    # 모델 생성
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # 훈련 세트 정확도 저장
    training_accuracy.append(clf.score(X_train, y_train))
    # 일반화 정확도 저장
    test_accuracy.append(clf.score(X_test, y_test))

#%%
plt.plot(neighbors_settings, training_accuracy, label="training_accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test_accuracy")
plt.ylabel("accuracy")
plt.xlabel("n_neighbors")
plt.legend()