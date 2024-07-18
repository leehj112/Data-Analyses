# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:35:11 2024

@author: leehj
"""

# KNN(k-Nearest-Neighbors)
# 최근접 이웃 분류 알고리즘

#%%
# pip install joblib
# pip install mglearn

#%%
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_breast_cancer
import mglearn
import matplotlib.pyplot as plt

# sklearn datasets: load_breast_cancer
cancer = load_breast_cancer() 
X_train, X_test, y_train, y_test = train_test_split(
   cancer.data, cancer.target, stratify=cancer.target, random_state=66)

#%%

import pandas as pd
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target

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
    # score() : 0-1의 범위를 벗어날 수 있다. 백분율(%) 환산
    training_accuracy.append(clf.score(X_train, y_train)) 
    # 일반화 정확도 저장
    test_accuracy.append(clf.score(X_test, y_test))

#%%
plt.plot(neighbors_settings, training_accuracy, label="training_accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test_accuracy")
plt.ylabel("accuracy")
plt.xlabel("n_neighbors")
plt.legend()