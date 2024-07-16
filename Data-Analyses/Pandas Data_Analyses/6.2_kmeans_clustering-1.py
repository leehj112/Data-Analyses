# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 12:12:49 2024

@author: leehj
"""


# 비지도학습(Unsupervised Learning)
# 군집(clustering) : k-Means, DBSCAN
# 분류와 차이는 정답이 없다.
# 유사성만을 기준으로 판단
# 신용카드 부정 사용 탐지, 구매 패턴 분석, 소비자 행동 특성

# K-평균 군집화(k-Means Clustering)
#   - 비지도 학습의 대표적인 알고리즘
#   - 목표 변수가 없는 상태에서 데이터를 비슷한 유형으로 분류
#   - K-최근접 알고리즘과 비슷하게 거리 기반(kNN)
#   - 전체 그룹의 수는 사용자가 지정한 K개
#   - 클러스터링은 데이터를 적절한 수의 그룹으로 나누어 그 특징을 살펴볼 수 있는 장점
#   - 중심점 K개를 임의로 설정
#   - 각 중심점을 기준으로 가까이에 있는 데이터를 해당 클러스터로 할당
#   - 더 이상 클러스터에 변동이 없을 때가지 분류를 반복
#   - 작동방식:
#     . 무작위로 k개의 클러스터 중심을 정함
#     . 각 샘플에서 가장 가까운 클러스터 중심을 찾아 해당 클러스터의 샘플로 지정
#     . 클러스터에 속한 샘플의 평균값으로 클러스터 중심을 변경
#     . 클러스터의 변화가 없을 때까지 반복   
#   - 장점: 간단하고 클러스터링 결과를 쉽게 해석
#   - 단점: 
#     . 최적의 K값을 사용자가 직접 선택
#     . 거리 기반 알고리즘이기 때문에 변수의 스케일에 따라 다른 결과를 나타남
#   - 적용분야:
#     . 마케팅, 제품기획 등을 목적으로 하는 고객 분류
#     . 탐색적 자료 분석, 피처 엔지니어링 용도로 사용 가능(*)
#   - 엘보우 기법
#     . 최적의 클러스터 개수를 확인하는 방법
#     . 클러스터의 중점과 각 데이터 간의 거리를 기반으로 계산
#   - 이너셔
#     . 각 클러스터의 중점과 그에 속한 데이터 간의 거리
#     . 값이 작을수록 잘 뭉쳐진 클러스터를 의미
#   - 실루엣 계수
#     . 엘보우 기법과 같이 최적의 클러스터 수를 찾는 방법
#     . 엘보우 기법에서 적절한 클러스터 수를 찾지 못했을 때 대안으로 사용
#     . 엘보우 기법보다 계산 시간이 오래 걸림
#     
#
# sklearn.cluster.KMeans()
#   - 예) sklearn.cluster.KMeans(init='k-means++', n_clusters=5, n_init=10)
#   - n_clusters: 기본값(8), 클러스터의 개수를 지정, 
#                 일반적으로 데이터의 특성을 잘 반영할 수 있는 적절한 클러스터 개수를 찾아야 함
#   - init: 초기 클러스터 중심점을 어떻게 설정할지 지정
#      . 'random' : 무작위로 초기화
#      . 'k-means++' : 더 나은 초기화 방법을 사용
#   - max_iter: 기본값(300), 최대 반복 횟수, 수렴할 때까지 더 많은 반복을 수행
#   - tol: 수렴 기준, 클러스터 중심점의 변화량이 이 값보다 작아지면 수렴한 것으로 간주
#   - random_state: 난수 생성기의 시드, 이 값을 고정하면 동일한 결과를 재현

#%%
### 기본 라이브러리 불러오기
import pandas as pd
import matplotlib.pyplot as plt


'''
[Step 1] 데이터 준비
'''

# Wholesale customers 데이터셋 가져오기 (출처: UCI ML Repository)
# https://archive.ics.uci.edu/dataset/292/wholesale+customers
# UCI 도소매업 고객 데이터셋
# 고객의 연간 구매금액을 상품 카테고리별로 구분하여 정리한 데이터

uci_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv'
df = pd.read_csv(uci_path, header=0)

df.to_csv("../datasets/wholesale.csv", index=False)

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


'''
[Step 3] 데이터 전처리
'''

# 분석에 사용할 속성을 선택
X = df.iloc[:, :]
print(X[:5])
print('\n')

#%%

# 설명 변수 데이터를 정규화
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)

print(X[:5])
print('\n')

#%%

'''
[Step 4] k-means 군집 모형 - sklearn 사용
'''

# sklearn 라이브러리에서 cluster 군집 모형 가져오기
from sklearn import cluster

# 모형 객체 생성 
kmeans = cluster.KMeans(init='k-means++', n_clusters=5, n_init=10)

#%%
# 모형 학습
kmeans.fit(X)   

#%%

# 예측 (군집) 
cluster_label = kmeans.labels_   
print(cluster_label)
print('\n')

#%%

# 예측 결과를 데이터프레임에 컬럼추가('Cluster')
df['Cluster'] = cluster_label
print(df.head())   
print('\n')

#%%

# 그래프로 표현 - 시각화(분포)
# Grocery(식료품), Frozen(냉동)
# df.plot(kind='scatter', x='Grocery', y='Frozen', c='Cluster', cmap='Set1', colorbar=False, figsize=(10, 10))
df.plot(kind='scatter', x='Grocery', y='Frozen', c='Cluster', cmap='Set1', colorbar=True, figsize=(10, 10))

# Milk(우유), Delicassen(델리)
df.plot(kind='scatter', x='Milk', y='Delicassen', c='Cluster', cmap='Set1', colorbar=True, figsize=(10, 10))
plt.show()
plt.close()

#%%
# 큰 값으로 구성된 클러스터(0, 4)를 제외 - 값이 몰려 있는 구간을 자세하게 분석
mask = (df['Cluster'] == 0) | (df['Cluster'] == 4)
ndf = df[~mask]

ndf.plot(kind='scatter', x='Grocery', y='Frozen', c='Cluster', cmap='Set1', colorbar=False, figsize=(10, 10))
ndf.plot(kind='scatter', x='Milk', y='Delicassen', c='Cluster', cmap='Set1', colorbar=True, figsize=(10, 10))
plt.show()
plt.close()