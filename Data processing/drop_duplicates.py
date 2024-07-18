# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:10:41 2024

@author: leehj
"""

# -*- coding: utf-8 -*-

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
# 데이터프레임에서 중복 행을 제거
df2 = df.drop_duplicates()
print(df2)
print('\n')

#%%
print(df)
df2 = df.drop_duplicates()
print(df2[['c2','c3']])
print('\n')

#%%
print(df2[['c2','c3']][2:4])

#%%
# c2, c3열을 기준으로 중복 행을 제거
print(df)
df3 = df.drop_duplicates(subset=['c2', 'c3'])
print(df3)