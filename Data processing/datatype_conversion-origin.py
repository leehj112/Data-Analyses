# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:12:07 2024

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

# origin: 출시국가, 카테고리 범주형
# 1:'USA', 2:'EU', 3:'JAPAN'

# origin 열의 고유값 확인 : unique()
origin_counts = df['origin'].value_counts()
origin_index = origin_counts.index
origin_values = origin_index.values # Numpy Array
print('origin_values:', origin_values)

#%%
print(pd.Series(df['origin'].value_counts().index).values) # [1 3 2]

#%%

print(df['origin'].drop_duplicates().values) # [1 3 2]

#%%
# 데이터에서 중복되는 자료를 제거
origin_unique = df['origin'].unique()
print(df['origin'].unique()) # [1 3 2]

#%%

# 정수형 데이터를 문자형 데이터로 변환 
df['origin'].replace({1:'USA', 2:'EU', 3:'JAPAN'}, inplace=True)

#%%
# origin 열의 고유값과 자료형 확인
print(df['origin'].unique())
print(df['origin'].dtypes) 
print('\n')

#%%
# origin 열의 문자열 자료형을 범주형으로 변환
df['origin'] = df['origin'].astype('category')     
print(df['origin'].dtypes) 

#%%
# 범주형을 문자열로 다시 변환
df['origin'] = df['origin'].astype('str')     
print(df['origin'].dtypes)

#%%
# model year 열의 정수형을 범주형으로 변환
print(df['model year'].sample(3))
df['model year'] = df['model year'].astype('category') 
print(df['model year'].sample(3)) 