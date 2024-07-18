# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:12:27 2024

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

# model year 열의 정수형을 범주형으로 변환
print(df['model year'].sample(3))

#%%

df['model year'] = df['model year'].astype('category') 

#%%
print(df['model year'].sample(3)) 