# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:36:05 2024

@author: leehj
"""

# -*- coding: utf-8 -*-
"""
[k-최근접 이웃 분류]
pip install mglearn
"""

from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_breast_cancer
import mglearn
import matplotlib.pyplot as plt

# mglearn.plots.plot_knn_classification(n_neighbors=1)
mglearn.plots.plot_knn_classification(n_neighbors=3)

X, y = mglearn.datasets.make_forge() # 임의의 샘플 데이터
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
print("테스트 세트 예측: {}".format(clf.predict(X_test)))
print("테스트 세트 정확도: {:.2f}".format(clf.score(X_test, y_test)))

#%%

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbors, ax in zip([1, 3, 9], axes):
    print("n_neighbors: ", n_neighbors)
    print("ax :", ax)
    # fit 메서드는 self 객체를 반환합니다.
    # 그래서 객체 생성과 fit 메서드를 한 줄에 쓸 수 있습니다.
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor".format(n_neighbors))
    ax.set_xlabel("specific 0")
    ax.set_ylabel("specific 1")
    
axes[0].legend(loc=3)