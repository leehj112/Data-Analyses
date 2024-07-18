# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:58:17 2024

@author: leehj
"""

# -*- coding: utf-8 -*-

# date_range() 함수
# 라이브러리 불러오기
import pandas as pd

# Timestamp의 배열 만들기 - 월 간격, 월의 시작일 기준
ts_ms = pd.date_range(start='2019-01-01',    # 날짜 범위의 시작
                   end=None,                 # 날짜 범위의 끝
                   periods=6,                # 생성할 Timestamp의 개수
                   freq='MS',                # 시간 간격 (MS: 월의 시작일)
                   tz='Asia/Seoul')          # 시간대(timezone)
print(ts_ms)
print('\n')

#%%
# 월 간격, 월의 마지막 날 기준
ts_me = pd.date_range('2019-01-01', 
                      periods=6, 
                      freq='ME',             # 시간 간격 (ME: 월의 마지막 날)
                      tz='Asia/Seoul')       # 시간대(timezone)
print(ts_me)
print('\n')

#%%
# 분기(3개월) 간격, 월의 마지막 날 기준
ts_3me = pd.date_range('2019-01-01', 
                      periods=6, 
                      freq='3ME',            # 시간 간격 (3ME: 3개월)
                      tz='Asia/Seoul')       # 시간대(timezone)
print(ts_3me)

#%%

# 분기초 : QS
ts_qs = pd.date_range('2023-01-01', 
                      periods=4, 
                      freq='QS',             # 시간 간격 (QS: 3개월)
                      tz='Asia/Seoul')       # 시간대(timezone)
print(ts_qs)

#%%

# 분기(3개월) 간격, 월의 시작 날 기준
ts_3ms = pd.date_range('2023-01-01', 
                      periods=4, 
                      freq='3MS',            # 시간 간격 (QS: 3개월)
                      tz='Asia/Seoul')       # 시간대(timezone)
print(ts_3ms)

#%%


# 분기말(Q) 월의 마지막 날 
ts_qs = pd.date_range('2023-01-01', 
                      periods=4, 
                      freq='QE',             # 시간 간격 (Q: 3개월)
                      tz='Asia/Seoul')       # 시간대(timezone)
print(ts_qs)