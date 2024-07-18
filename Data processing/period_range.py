# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:58:55 2024

@author: leehj
"""

# -*- coding: utf-8 -*-

# 라이브러리 불러오기
import pandas as pd

# Period 배열 만들기 - 1개월 길이
pr_m = pd.period_range(start='2019-01-01',     # 날짜 범위의 시작
                   end=None,                   # 날짜 범위의 끝
                   periods=3,                  # 생성할 Period 개수
                   freq='M')                   # 기간의 길이 (M: 월)
print(pr_m)
print('\n')

#%%
# Period 배열 만들기 - 1시간 길이
pr_h = pd.period_range(start='2019-01-01',     # 날짜 범위의 시작
                   end=None,                   # 날짜 범위의 끝
                   periods=3,                  # 생성할 Period 개수
                   freq='h')                   # 기간의 길이 (h: 시간)
print(pr_h)
print('\n')

#%%

# Period 배열 만들기 - 1시간 길이
pr_h2 = pd.period_range(start='2023-09-07 09:00:00',     # 날짜 범위의 시작
                   end=None,                   # 날짜 범위의 끝
                   periods=4,                  # 생성할 Period 개수
                   freq='h')                   # 기간의 길이 (h: 시간)
print(pr_h2)
print('\n')



#%%
# Period 배열 만들기 - 2시간 길이
pr_2h = pd.period_range(start='2019-01-01',    # 날짜 범위의 시작
                   end=None,                   # 날짜 범위의 끝
                   periods=3,                  # 생성할 Period 개수
                   freq='2h')                  # 기간의 길이 (h: 시간)
print(pr_2h)