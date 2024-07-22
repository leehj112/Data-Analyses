# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:36:03 2024

@author: leehj
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 09:38:55 2024

@author: Solero
"""

#%%

# [PCA, Principal Component Analysis]

"""
1. 주성분 분석
2. 차원축소
3. 비지도 학습에 속하기 때문에 종속변수는 존재하지 않으며 어떤 것을 예측하지도 분류하지도 않음
4. PCA의 목적은 데이터의 차원을 축소하는 데 있다.
5. 차원축소
   - 변수의 갯수를 줄이되, 가능한 그 특성을 보존해 내는 기법(변수 압축)
   - 기존의 변수 중 일부를 그대로 선택하는 방식이 아니라,
     기존 변수들의 정보를 모두 반영하는 새로운 변수들을 만드는 방식으로 차원을 축소
   - 변수가 두 개면 2차원 그래프로, 세 개면 3차원 그래프로 나타낼 수 있음
     데이터의 차원은 변수의 갯수와 직결
6. 장점
   - 다차원을 2차원에 적합하도록 차원 축소하여 시각화에 유용
   - 변수 간의 높은 상관관계 문제를 해결
7. 단점
   - 기본 변수가 아닌 새로운 변수를 사용하여 해석하는데 어려움이 있음
   - 차원이 축소됨에 따라 정보 손실이 발생
8. 유용한 곳
   - 다차원 변수들을 2차원 그래프로 표현하는 데 사용
   - 변수가 너무 많아 모델 학습에 시간이 많이 걸릴 때
   - 오버피팅을 방지하는 용도로 사용
"""

#%%

# 데이터셋: [09]_practice-2-pca.csv
# 스케일링 된 고객별 총 지출금액 및 범주별 지출금액
# 각 고객이 속한 클러스터 라벨(label)

#%%

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

#%%
# 데이터 셋 읽기
customer = pd.read_csv("./[09]_practice-2-pca.csv")

# 독립변수 지정
customer_X = customer.drop('label', axis=1)

# 종속변수 지정
customer_y = customer['label']


#%%
# 주성분 개수 지정
pca = PCA(n_components=2)

# 학습
pca.fit(customer_X)

# 변환
# 결과 : 넘파이 배열
pca_X = pca.transform(customer_X)

# 데이터프레임으로 변환
pca_df = pd.DataFrame(pca_X, columns=['pca1', 'pca2'])

# 타겟 데이터 합치기
# 타겟 레이블: y
pca_df = pca_df.join(customer_y)

#%%

print(pca_df)

#%%

# 산점도 그래프 그리기
sns.scatterplot(x='pca1', y='pca2', data=pca_df, hue='label', palette='rainbow')

#%%
# 주성분과 변수의 관계 확인
print(pca.components_)

#%%
# 데이터프레임으로 변환
df_comp = pd.DataFrame(pca.components_, columns=customer_X.columns)

#%%

# 히트맵 그리기
# 양수: 빨란색
# 음수: 파란색
sns.heatmap(df_comp, cmap='coolwarm')

#%%

# THE END