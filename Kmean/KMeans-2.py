# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 09:42:27 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# K-평균 군집화(K-means Clustering)
# 비지도학습의 대표적인 알고리즘으로 목표변수가 없는 상태에서 데이터를 비슷한 유형끼리 묶는 머신러닝 기법이다.
# K-최근접 이웃 알고리즘과 비슷하게 거리 기반으로 작동하며 적절한 K값을 사용자가 지정해야 한다.
# 거리 기반으로 작동하기 때문에 데이터 위치가 가까운 데이터끼리 한 그룹으로 묶는다.
# 전체 그룹의 수는 사용자가 지정한 K개이다.
#
# 장점:
#   - 구현이 간단하다.
#   - 클러스터링 결과를 쉽게 해석할 수 있다.

# 단점:
#   - 최적의 K값을 자동으로 찾지 못하고 사용자가 직접 선택해야 한다.
#   - 거리 기반 알고리즘이기 때문에 변수의 스케일에 따라 다른 결과를 나타낼 수 있다.
#
# 유용한 곳:
#   - 종속변수가 없는 데이터셋에서 데이터 특성을 비교적 간단하게 살펴보는 용도로 활용할 수 있다.
#   - 마케팅이나 제품 기획 등을 목적으로 한 고객 분류에 사용할 수 있다.
#   - 지도 학습에서 종속변수를 제외하면, 탐색적 자료 분석 혹은 피처 엔지니어링 용도로 사용할 수 있다.     
#
# 시나리오:
#   - 온라인 쇼핑몰의 고객이 구매한 물품, 검색한 물품, 살펴본 물품 정보를 이용해 고객에게 추천 서버스를 제공
#   - 다양한 변수를 활용하여 다양한 방식으로 고객 그룹을 분류
#   - 고객 데이터셋을 분석하여 적당한 수의 그룹으로 묶어보고
#     각 그룹별로 어떤 특성이 있는지 알아 봄    
#
# 문제유형
#   - 비지도학습
# 
# 평가지표:
#   - 엘보우 기법
#   - 실루엣 점수    

#%%

###############################################################################
# 엘보우 기법으로 최적의 K값 구하기
##############################################################################
# 엘보우 기법(elbow method)
#   - 최적의 클러스터 개수를 확인하는 방법
#   - 클러스터의 중점과 각 데이터 간의 거리를 기반으로 계산
# 이너셔(inertia)
#   - 각 그룹에서의 중심과 각 그룹에 해당하는 데이터 간의 거리에 대한 합
#   - 작을수록 그룹별로 더 오밀조밀 잘 모이게 분류됐다고 할 수 있다.
#   - K 값이 커지면 거리의 합인 이너셔는 필연적으로 작아지게 된다.
#   - 유의미한 클러스터를 만들어서 어떠한 집단적 특징을 찾는 것이 목적

#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

#%%

###############################################################################
# 고객 데이터셋
###############################################################################
# cc_num   : 카드번호
# category : 범주
# amt      : 거래금액
###############################################################################

#%%

file_url = '../dataset/[ML]_KMeans_customer.csv'
customer = pd.read_csv(file_url)

#%%

customer.head()

#%%

# 카드 고유값 확인
# 고객 100에 대한 카드 사용 정보
customer['cc_num'].nunique() # 100

#%%

# 범주: 11개
customer['category'].nunique() # 11

#%%

###############################################################################
# 전처리 : 피처 엔지니어링
###############################################################################

#%%

# 더미 변수로 변환
customer_dummy = pd.get_dummies(customer, columns =['category'])

#%%

customer_dummy.head()

#%%

# 변수 이름 리스트 : 11개
cat_list = customer_dummy.columns[2:]

#%%

# 범주별로 사용한 금액을 계산
# True인 위치에 금액을 넣음
for n in cat_list:
    customer_dummy[n] = customer_dummy[n] * customer_dummy['amt']


#%%

customer_dummy

#%%

# 고객 레벨로 취합 : groupby() 함수 사용
# 고객별 총 사용 금액 및 범주별 사용 금액
customer_agg = customer_dummy.groupby('cc_num').sum()

#%%

customer_agg.head()

#%%

# 스케일링 작업: StandardScaler
# K-평균 군집화는 거리 기반 알고리즘이기 때문에 데이터의 스케일에 영향을 받음
# 
# 다른 고객들과 비슷한 수준이면, 즉 평균에 가까울 경우 0에 근접한 값
# 더 많이 사용했으면 더 큰 양수
# 더 적게 사용했으면 더 작은 음수

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()  

scaled_df = pd.DataFrame(scaler.fit_transform(customer_agg), 
            columns = customer_agg.columns, 
            index=customer_agg.index) 

#%%

scaled_df.head()


#%%

###############################################################################
# 고객 데이터 모델링 및 실루엣 계수
###############################################################################

#%%

# 엘보우 기법으로 K값 확인
distance = []
for k in range(2,10):
    k_model = KMeans(n_clusters=k)
    k_model.fit(scaled_df)
    labels = k_model.predict(scaled_df)
    distance.append(k_model.inertia_)

#%%

# 엘보우 플랏
sns.lineplot(x=range(2,10), y=distance)

#%%

# 엘보우 기법 결과:
#   - 한 저점에서 크게 떨어지지 않고 비교적 완만하게 그래프가 내려가고 있다.
#   - K값을 결정하기 어려운 모양이다.

#%%

###############################################################################
# 고객 데이터 모델링 및 실루엣 계수
###############################################################################
# 실루엣 계수(silhouette coefficient)
#   - 엘보우 기법과 같이 최적의 클러스터 수를 찾는 방법
#   - 엘보우 기법에서 적절한 클러스터 수를 찾지 못했을 때 대안으로 사용
#   - 엘보우 기법보다 계산 시간이 오래 걸린다.
#   - 클러스터 내부에서의 평균 거리와, 최근접한 다른 클러스터 데이터와의 평균 거리도 반영


#%%

###############################################################################
# 실루엣 점수
from sklearn.metrics import silhouette_score

#%%

# 실루엣 계수를 저장할 리스트
silhouette = []

for k in range(2,10):
    k_model = KMeans(n_clusters=k)      # 모델객체 생성
    k_model.fit(scaled_df)              # 학습
    labels = k_model.predict(scaled_df) # 예측
    silhouette.append(silhouette_score(scaled_df, labels)) # 실루엣 계수 추가

#%%

silhouette

#%%

# 실루엣 계수가 높은 값일수록 더 좋은 분류를 의미
sns.lineplot(x=range(2,10), y=silhouette)

#%%

###############################################################################
# 최종 예측 모델 및 결과 해석
###############################################################################

#%%

k_model = KMeans(n_clusters=4)
k_model.fit(scaled_df)
labels = k_model.predict(scaled_df)

#%%

# 정답을 데이터프레임에 붙임
scaled_df['label'] = labels

#%%

# 레이블(label)별 평균값
scaled_df_mean = scaled_df.groupby('label').mean() 
scaled_df_mean

#%%

# 레이블(label)별 등장횟수
scaled_df_count = scaled_df.groupby('label').count()['category_travel'] 
scaled_df_count

#%%

# 컬럼이름 변경
# category_travel -> count
scaled_df_count = scaled_df_count.rename('count')
scaled_df_count

#%%

scaled_df_all = scaled_df_mean.join(scaled_df_count)

#%%


scaled_df_all

#%%

# THE END