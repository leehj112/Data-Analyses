# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:07:28 2024

@author: leehj
"""

# -*- coding: utf-8 -*-

# 누락 데이터 치환 : fillna()

# 라이브러리 불러오기
import seaborn as sns

# titanic 데이터셋 가져오기
df = sns.load_dataset('titanic')

# age 열의 첫 10개 데이터 출력 (5 행에 NaN 값)
print(df['age'].head(10))
print('\n')

#%%
# age 열의 NaN값을 다른 나이 데이터의 평균으로 변경하기
#mean_age = df['age'].mean(axis=0)   # age 열의 평균 계산 (NaN 값 제외)
mean_age = df[['age','fare']].mean(axis=0)   # age 열의 평균 계산 (NaN 값 제외)
mean_age_fare = mean_age.mean() # age, fare의 평균
df['age'].fillna(mean_age_fare, inplace=True)

#%%
# age 열의 첫 10개 데이터 출력 (5 행에 NaN 값이 평균으로 대체)
print(df['age'].head(10))

#%%
