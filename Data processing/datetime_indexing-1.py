# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:59:51 2024

@author: leehj
"""

# 라이브러리 불러오기
import pandas as pd

# read_csv() 함수로 파일 읽어와서 df로 변환
df = pd.read_csv('stock-data.csv')

# 문자열인 날짜 데이터를 판다스 Timestamp로 변환
df['new_Date'] = pd.to_datetime(df['Date'])   # 새로운 열에 추가
df.set_index('new_Date', inplace=True)        # 행 인덱스로 지정

#%%

# 내림차순
df_ymd_range = df[
        pd.to_datetime('2018-06-25') :
        pd.to_datetime('2018-06-20')]    # 날짜 범위 지정
print(df_ymd_range)
print('\n')

#%%

# 오름차순 : 데이터가 검색되지 않음
df_ymd_range = df[  # 날짜 범위 지정
        pd.to_datetime('2018-06-20') :   
        pd.to_datetime('2018-06-25') ]
print(df_ymd_range)
print('\n')

#%%

# 오류
# 날짜 범위 지정
# print(df['2018-06-25' : '2018-06-20'])
# print(df.loc['2018-06-25' : '2018-06-20'])