# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:57:49 2024

@author: leehj
"""

# -*- coding: utf-8 -*-

# 문자열(날짜)를 pandas.datetime64 자료형으로 변환
# pd.to_datetime(df['Date']) 

# 라이브러리 불러오기
import pandas as pd

# read_csv() 함수로 CSV 파일을 가져와서 df로 변환
df = pd.read_csv('stock-data.csv')

# 데이터 내용 및 자료형 자료형 확인
print(df.head())
print('\n')
print(df.info())

#%%
# 문자열 데이터(시리즈 객체)를 판다스 Timestamp로 변환
df['new_Date'] = pd.to_datetime(df['Date'])   #df에 새로운 열로 추가

#%%
# 데이터 내용 및 자료형 자료형 확인
print(df.head())
print('\n')
#%%
print(df.info())
print('\n')
#%%
print('type:', type(df['new_Date'][0])) # <class 'pandas._libs.tslibs.timestamps.Timestamp'>

#%%
# 시계열 값으로 변환된 열을 새로운 행 인덱스로 지정. 기존 날짜 열은 삭제
df.set_index('new_Date', inplace=True)
df.drop('Date', axis=1, inplace=True)

#%%
# 데이터 내용 및 자료형 자료형 확인
print(df.head())
print('\n')
print(df.info())