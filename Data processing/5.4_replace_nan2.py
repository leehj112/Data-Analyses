# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:08:19 2024

@author: leehj
"""

# -*- coding: utf-8 -*-

# 라이브러리 불러오기
import seaborn as sns

# titanic 데이터셋 가져오기
df = sns.load_dataset('titanic')

# 승객들이 승선한 경유 도시
# embark_town 열의 829행의 NaN 데이터 출력
print(df['embark_town'][825:830]) # 829 NaN
print('\n')

#%%
# embark_town 열의 NaN값을 승선도시 중에서 가장 많이 출현한 값으로 치환하기
# most_freq = df['embark_town'].value_counts(dropna=True).idxmax()   
print(df['embark_town'].value_counts(dropna=False)) # NaN:2

# NaN을 포함해서 카운트를 함, 결과 NaN 2건으로 가장 작음
# 그러므로 리턴은 nan
print(df['embark_town'].value_counts(dropna=False).idxmin()) # nan

# NaN을 제외하면 Queenstown이 77건으로 가장 작은 값의 칼럼
print(df['embark_town'].value_counts(dropna=True).idxmin()) # Queenstown

#%%
most_freq = df['embark_town'].value_counts().idxmax()   
print(most_freq)
print('\n')

#%%
df['embark_town'].fillna(most_freq, inplace=True)

# embark_town 열 829행의 NaN 데이터 출력 (NaN 값이 most_freq 값으로 대체)
print(df['embark_town'][825:830])