# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:05:50 2024

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
print(missing_count)

#%%
"""
age
False    714  # 유효한 갯수
True     177  # 무효한 갯수
Name: count, dtype: int64
"""
#%%

# 데이터셋의 모든 컬럼별로 NaN 갯수를 출력
missing_df = df.isnull()

for col in missing_df.columns:
    missing_count = missing_df[col].value_counts()    # 각 열의 NaN 개수 파악

    try: 
        print(col, ': ', missing_count[True])   # NaN 값이 있으면 개수를 출력
    except: # key(True)가 없으면 KeyError: True
        print(col, ': ', 0)                     # NaN 값이 없으면 0개 출력
        

#%%

print(len(df.columns), df.columns) # 15

#%%

# 500개 이상 NaN을 가지고 있는 컬럼은 삭제
# deck :  688, isnull
df_thresh = df.dropna(axis=1, thresh=500)
print(len(df_thresh.columns), df_thresh.columns) # 14

#%%

# axis = 0인 경우?
# 결측값이 1개 이상 있는 행(로우) 삭제
df_droped_rows = df.dropna(axis=0)  
print(df_droped_rows)

#%%

help(df.dropna)

#%%

# axis : 0('index', 'rows'), 1(columns)
# 기본값: 0
df_droped_rows = df.dropna()  
print(df_droped_rows)

#%%

df_droped_rows = df.dropna(axis='rows')  
print(df_droped_rows)

#%%
"""
dropna() 옵션
how : {'any', 'all'}, default 'any'
    Determine if row or column is removed from DataFrame, when we have
    at least one NA or all NA.

    * 'any' : If any NA values are present, drop that row or column.
    * 'all' : If all values are NA, drop that row or column.
"""

#%%

# how:
#   - 'any', 1개 이상의 값이 NA이면, 기본값
#   - 'all', 모든 값이 NA이면

#%%

df_droped_any = df.dropna(how='any')
print(df_droped_any) # 182

#%%

# 모든 컬럼이 NA이면 해당하는 행을 삭제
df_droped_all = df.dropna(how='all')
print(df_droped_all)

#%%

# 지정한 컬럼에 결측값이 있는 행만 삭제
# 지정한 컬럼에서 하나라도 결측값이 있는 행이 삭제
del_cols = ['age', 'embarked']
df_droped_subs = df.dropna(subset=del_cols)
print(df_droped_subs) # 712


#%%

# 특정한 컬럼의 범위를 지정
del_cols = df.columns[3:8]
df_droped_subs = df.dropna(subset=del_cols)
print(df_droped_subs) # 712