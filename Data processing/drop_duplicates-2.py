# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:10:25 2024

@author: leehj
"""

# 중복제거
# df.drop_duplicates()
# df.drop_duplicates(subset=['c2', 'c3'])

# 라이브러리 불러오기
import pandas as pd

# 중복 데이터를 갖는 데이터프레임 만들기
df = pd.DataFrame({'c1':['a', 'a', 'b', 'a', 'b'],
                  'c2':[1, 1, 1, 2, 2],
                  'c3':[1, 1, 2, 2, 2]})
print(df)
print('\n')

#%%
# c2, c3열을 기준으로 중복 행을 제거
print(df[['c2', 'c3']])

# 리턴 : 'c2', 'c3'에서 중복을 제거 한 결과
df3 = df[['c2', 'c3']].drop_duplicates()
print(df3)