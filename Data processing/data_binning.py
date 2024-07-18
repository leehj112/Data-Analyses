# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:31:07 2024

@author: leehj
"""

# -*- coding: utf-8 -*-

# 구간분할
# pd.cut(...)
# 구간범위: (max - min) / bins

# 라이브러리 불러오기

import pandas as pd
import numpy as np

# read_csv() 함수로 df 생성
df = pd.read_csv('./auto-mpg.csv', header=None)

# 열 이름을 지정
df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name'] 

df.info()
#%%

df['horsepower'].value_counts()

#%%
# FutureWarning: A value is trying to be set on a copy of a DataFrame 
# or Series through chained assignment using an inplace method.

# horsepower 열의 누락 데이터('?') 삭제하고 실수형으로 변환
df['horsepower'].replace('?', np.nan, inplace=True)      # '?'을 np.nan으로 변경

#%%

# 권고
df['horsepower'] = df['horsepower'].replace({'?': np.nan})      # '?'을 np.nan으로 변경


#%%

df.dropna(subset=['horsepower'], axis=0, inplace=True)   # 누락데이터 행을 삭제
df['horsepower'] = df['horsepower'].astype('float64')      # 문자열을 실수형으로 변환

df.info()
#%%

# np.histogram 함수로 3개의 bin으로 나누는 경계 값의 리스트 구하기
# 구간범위: max - min / bins
# [ 46. 107.33333333 168.66666667 230. ]
# 46~107.3, 107.3~168.6, 168.6~230
horsepower_min = df['horsepower'].min()
horsepower_max = df['horsepower'].max()
horsepower_bin = (horsepower_max - horsepower_min) / 3
print("horsepower_min: ", horsepower_min)
print("horsepower_max: ", horsepower_max)
print("horsepower_bin: ", horsepower_bin)

count, bin_dividers = np.histogram(df['horsepower'], bins=3)
print("count:", count) 
print(bin_dividers) 

#%%

# 3개의 bin에 이름 지정
bin_names = ['저출력', '보통출력', '고출력']

# pd.cut 함수로 각 데이터를 3개의 bin에 할당
df['hp_bin'] = pd.cut(x=df['horsepower'],     # 데이터 배열
                      bins=bin_dividers,      # 경계 값 리스트
                      labels=bin_names,       # bin 이름
                      include_lowest=True)    # 첫 경계값 포함 

# horsepower 열, hp_bin 열의 첫 15행을 출력
print(df[['horsepower', 'hp_bin']].head(15))