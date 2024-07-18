# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:58:08 2024

@author: leehj
"""

# -*- coding: utf-8 -*-

# 라이브러리 불러오기
import pandas as pd

# 날짜 형식의 문자열로 구성되는 리스트 정의
dates = ['2019-01-01', '2020-03-01', '2021-06-01']

#%%
# 문자열 데이터(시리즈 객체)를 판다스 Timestamp로 변환
ts_dates = pd.to_datetime(dates)   
print(ts_dates)
print('\n')

#%%
# Timestamp를 Period로 변환
pr_day = ts_dates.to_period(freq='D') # 년-월-일
print(pr_day)

#%%
pr_month = ts_dates.to_period(freq='M') # 년-월
print(pr_month)

#%%

# pr_year = ts_dates.to_period(freq='A') # 연도(Annual)
pr_year = ts_dates.to_period(freq='Y') # 연도(Year)
print(pr_year)