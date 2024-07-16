# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:15:39 2024

@author: leehj
"""

# 라이브러리 불러오기
import pandas as pd

# 데이터셋 가져오기
df = pd.read_excel('./주가데이터.xlsx', engine= 'openpyxl')
print(df.head(), '\n')
print(df.dtypes, '\n')

#%%
# 연, 월, 일 데이터 분리하기
# 숫자로 처리

# dates = df['연월일']   # Series
dates = pd.to_datetime(df['연월일'])

df_y = dates.dt.year   # Series
df_m = dates.dt.month  # Series
df_d = dates.dt.day    # Series

#%%
# 분리된 정보를 각각 새로운 열에 담아서 df에 추가하기
df['연'] = df_y
df['월'] = df_m
df['일'] = df_d
print(df.head())

#%%

print(df.info())