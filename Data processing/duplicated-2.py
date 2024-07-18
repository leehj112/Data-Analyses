# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:09:32 2024

@author: leehj
"""

# -*- coding: utf-8 -*-

# 라이브러리 불러오기
import pandas as pd

# 중복 데이터를 갖는 데이터프레임 만들기
df = pd.DataFrame({'c1':['a', 'a', 'b', 'a', 'b'],
                  'c2':[1, 1, 1, 2, 2],
                  'c3':[1, 1, 2, 2, 2]})
print(df)
print('\n')


#%%
# 데이터프레임`의 여러개의 열을 묶어서 중복값 찾기
print(df)
col_dup = df[['c2','c3']].duplicated()
print(col_dup)

#%%


