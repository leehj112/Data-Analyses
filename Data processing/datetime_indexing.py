# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:05:24 2024

@author: leehj
"""

# 라이브러리 불러오기
import pandas as pd

# read_csv() 함수로 파일 읽어와서 df로 변환
df = pd.read_csv('./stock-data.csv')

# 문자열인 날짜 데이터를 판다스 Timestamp로 변환
df['new_Date'] = pd.to_datetime(df['Date'])   # 새로운 열에 추가
df.set_index('new_Date', inplace=True)        # 행 인덱스로 지정

#%%
print(df.head())
print('\n')
print(df.index)
print('\n')

#%%

df.info()

#%%
# 날짜 인덱스를 이용하여 데이터 선택하기
# 오류 : KeyError: '2018'
# df_y = df['2018'] 
# print(df_y.head())

#%%
df_y = df.loc['2018']
print(df_y)
print('\n')

#%%
df_ym = df.loc['2018-07']    # loc 인덱서 활용
print(df_ym)
print('\n')

#%%
df_ym_cols = df.loc['2018-06', 'Start':'Volume']    # 열 범위 슬라이싱
print(df_ym_cols)
print('\n')

#%%

# KeyError: '2018-07-02'
# df_ymd = df['2018-07-02']

#%%

sr_ymd = df.loc['2018-07-02']
print(sr_ymd) # Series
print('\n')

#%%

df_ymd = df.loc[['2018-07-02']]
print(df_ymd) # DataFrame
print('\n')


#%%

# 인덱스에서 개별적인 데이터를 추출
df_y1 = pd.to_datetime('2018-06-05')
df_y2 = pd.to_datetime('2018-06-29')
print(df_y1, df_y2)

df_yy = df.loc[[df_y1,df_y2]] # 인덱스에서 개별적으로 조회
print(df_yy)

#%%

# 슬라이싱(범위) 
# 문제점: 인덱스가 날짜형인 경우 슬라이싱 방법?

# 인덱스를 정렬해야 함
# 오름차순 정렬을 해야 함
adf = df.sort_index()

# 오름차순 
# 시작보다 종료값이 크게 지정
adf_ymd_range = adf.loc[pd.to_datetime('2018-06-20'):pd.to_datetime('2018-06-25')]    # 날짜 범위 지정
print(adf_ymd_range)
print('\n')

#%%

# 내림차순으로 정렬
ddf = df.sort_index(ascending=False)

# 시작날짜가 종료날짜보다 커야 한다.
ddf_ymd_range = ddf.loc[pd.to_datetime('2018-06-25'): pd.to_datetime('2018-06-20')]    # 날짜 범위 지정
print(ddf_ymd_range)

#%%

# 시간 간격 계산. 최근 180일 ~ 189일 사이의 값들만 선택하기

ndf = df.copy()

today = pd.to_datetime('2018-12-25')            # 기준일 생성
ndf['time_delta'] = today - df.index            # 날짜 차이 계산
ndf.set_index('time_delta', inplace=True)       # 행 인덱스로 지정

df_180 = ndf.loc['180 days':'189 days']
print(df_180)