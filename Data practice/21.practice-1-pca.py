# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 23:50:24 2024

@author: dlgur
"""

#!/usr/bin/env python
# coding: utf-8

# 차원축소와 주성분 분석(PCA)
# PCA(Principal Component Analysis)

# 차원의 저수
#   - 독립변수가 늘어날수록 필요한 학습 데이터량이 기하급수적으로 늘어나는 상황
#   - 독립변수가 늘어난 만큼 충분한 데이터량이 뒷받침되지 못하면
#     . 과적합 문제 발생
#     . 다중공선성 문제 발생

# 차원축소
#   - 상관분석을 통해 독립변수 중에서 종속변수와 상관계수가 높은 독립변수만을 선택


# 다차원 척도법(MDS, Multi-Demensional Scaling)
#   - 데이터의 고차원 공간에서의 유사성 또는 거리 정보를 저차원 공간으로 투영하여 시각적으로 표현하는 기법
#   - 복잡한 데이터 구조를 2차원 또는 3차원 공간에 투영함으로써 데이터 간의 관계를 보다 쉽게 이해

# 주성분 분석(PCA)
#   - 기존의 독립변수들을 결합해 새로운 독립변수를 만들어(Feature Extraction) 모델링
#   - 여러 개의 독립변수를 서로 상관성이 높은 변수들의 선형 결합으로 만드는 방법
#   - 다양한 변수(고차원)로 인해 설명이 어려운 데이터를 축소된 차원을 통해 시각화시켜 보다 쉽게 설명
#   - 상관성이 높은 변수들의 선형 결합으로 새로운 변수가 만들어지기 때문에 가중공선성 문제에서 자유로워지며 속도 향상

#%%

# iris 데이터 셋을 이용해서 주성분 분석
# 품종별 특징을 나타내는 4개의 연속형 독립변수를 몇 개의 주성분으로 줄일 수 있는지 확인
# 변수: sepal_length,  sepal_width.  petal_length,  petal_width, species

import numpy as np
import pandas as pd


#%%

# iris 데이터셋(데이터프레임) 불러오기
import seaborn as sns
iris = sns.load_dataset('iris')
iris.head()

#%%

# species만 제외하고, 별도 데이터셋 구성
iris_n = iris.iloc[:,0:4].values

#%%

# 주성분 분석은 스케일의 영향을 받기 때문에 동일 스케일로 변환이 필요
# Standard Scaling 
from sklearn.preprocessing import StandardScaler
iris_s = StandardScaler().fit_transform(iris_n)


#%%

# 주성분 4개로 학습 실시
from sklearn.decomposition import PCA
pca4 = PCA(n_components=4).fit(iris_s)


#%%

# 주성분 4개까지 기여율 표시
pca4.explained_variance_ratio_

# 결과: array([0.72962445, 0.22850762, 0.03668922, 0.00517871])

# 독립변수 기여율
# 약 96% = 제1주성분(73%) + 제2주성분(23%)
# 주성분 분석 결과에서 적정한 주성분 개수를 선택하기 위한 판단 기준으로 
# 누적 기여율(Cumulative Proportion)이 85% 이상이거나 
# Scree Plot을 그렸을 때 수평을 유지하기 전 단계까지르 주성분의 개수로 선택

#%%

# scree plot 통해 적정 주성분 개수 확인
import matplotlib.pyplot as plt
plt.figure(figsize = (10,7))
PC_values = np.arange(pca4.n_components_) + 1
plt.plot(PC_values, pca4.explained_variance_ratio_, 'o-', color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

#%%

# 주성분별 선형결합 계수 출력
pca4.components_


#%%

# 주성분 2개로 학습 실시
pca2 = PCA(n_components=2).fit(iris_s)


#%%

# 스케일된 iris 데이터셋 주성분 2개로 변환
iris_pca = pca2.transform(iris_s)


#%%

# 주성분 2개로 신규 iris 데이터프레임 만들기
iris_res = pd.DataFrame(data = iris_pca, columns = ['comp1','comp2'])
iris_res['species'] = iris.species
iris_res.head()


#%%

# 주성분 2개로 산점도 그리기
import matplotlib.pyplot as plt
plt.figure(figsize = (10,7))
sns.scatterplot(data = iris_res, x='comp1', y='comp2', hue='species', style = 'species')
plt.show()

#%%

# THE END