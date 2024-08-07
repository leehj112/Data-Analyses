# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:12:45 2024

@author: leehj
"""

# -*- coding: utf-8 -*-

# 라이브러리 불러오기
import pandas as pd

# read_csv() 함수로 df 생성
df = pd.read_csv('./auto-mpg.csv', header=None)

# 열 이름을 지정
df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name'] 

# 각 열의 자료형 확인
print(df.dtypes)   
print('\n')

#%%

print(df['horsepower'])

#%%
# horsepower 열의 고유값 확인 : 마력
# Numpy object array
horsepower_unique = df['horsepower'].unique() # '?' 데이터도 출력

print(df['horsepower'].unique()) # '?' 데이터도 출력
print('\n')

#%%
# 누락 데이터('?') 삭제 
import numpy as np
df['horsepower'].replace('?', np.nan, inplace=True)      # '?'을 np.nan으로 변경
df.dropna(subset=['horsepower'], axis=0, inplace=True)   # 누락데이터 행을 삭제

horsepower_type = df['horsepower'].astype('float')

# 자료형을 변경시킨 시리즈 데이터로 변경
df['horsepower'] = df['horsepower'].astype('float')      # 문자열을 실수형으로 변환

#%%
# horsepower 열의 자료형 확인
print(df['horsepower'].dtypes)  
print('\n')

#%%
# origin 열의 고유값 확인
print(df['origin'].unique())

# 정수형 데이터를 문자형 데이터로 변환 
df['origin'].replace({1:'USA', 2:'EU', 3:'JAPAN'}, inplace=True)

# origin 열의 고유값과 자료형 확인
print(df['origin'].unique())
print(df['origin'].dtypes) 
print('\n')

# origin 열의 문자열 자료형을 범주형으로 변환
df['origin'] = df['origin'].astype('category')     
print(df['origin'].dtypes) 

# 범주형을 문자열로 다시 변환
df['origin'] = df['origin'].astype('str')     
print(df['origin'].dtypes)

# model year 열의 정수형을 범주형으로 변환
print(df['model year'].sample(3))
df['model year'] = df['model year'].astype('category') 
print(df['model year'].sample(3)) 