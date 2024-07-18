# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:06:15 2024

@author: leehj
"""

# -*- coding: utf-8 -*-

# 누락 데이터 제거
# df_thresh = df.dropna(axis=1, thresh=500)  

# 라이브러리 불러오기
import seaborn as sns

# titanic 데이터셋 가져오기
df = sns.load_dataset('titanic')

#%%
# for 반복문으로 각 열의 NaN 개수 계산하기
# 유효하지 않은 데이터는 True, 유효한 데이터는 False
missing_df = df.isnull()

col = 'age'
missing_count = missing_df[col].value_counts()    # 각 열의 NaN 개수 파악

#%%
# 'deck'
col = 'deck'
missing_count = missing_df[col].value_counts()    # 각 열의 NaN 개수 파악

#%%

# 'who'
col = 'who'
missing_count = missing_df[col].value_counts()    # 각 열의 NaN 개수 파악
print(col, ': ', missing_count[True])   # NaN 값이 있으면 개수를 출력

#%%

missing_df = df.isnull()

for col in missing_df.columns:
    missing_count = missing_df[col].value_counts()    # 각 열의 NaN 개수 파악

    try: 
        print(col, ': ', missing_count[True])   # NaN 값이 있으면 개수를 출력
    except: # key(True)가 없으면 KeyError: True
        print(col, ': ', 0)                     # NaN 값이 없으면 0개 출력

#%%
# NaN 값이 500개 이상인 열을 모두 삭제 - deck 열(891개 중 688개의 NaN 값)
df_thresh = df.dropna(axis=1, thresh=500)  
print(df_thresh.columns)

#%%

# age 열에 나이 데이터가 없는 모든 행을 삭제
# age 열(891개 중 177개의 NaN 값)
# how = 'any' : 하나라도 존재하면 삭제
# 칼럼이 하나이면 'any'나 'all'을 차이가 없다.
df_age = df.dropna(subset=['age'], how='any', axis=0)  
print(len(df_age))

#%%
# how = 'all' : 모든 데이터가 NaN이면 삭제
df_age_deck = df.dropna(subset=['age','deck'], how='all', axis=0)  
print(len(df_age))

#%%

# how = 'any' : 'age'와 'deck' 중에 하나라도 NaN이면 삭제
df_age_deck = df.dropna(subset=['age','deck'], how='any', axis=0)  
print(len(df_age))