# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:13:57 2024

@author: leehj
"""

# 라이브러리 불러오기
import seaborn as sns

# titanic 데이터셋의 부분을 선택하여 데이터프레임 만들기
titanic = sns.load_dataset('titanic')
df = titanic.loc[0:4, 'survived':'age']
print(df, '\n')

#%%
# 열 이름의 리스트 만들기
df_columns = df.columns
print("df.columns:", df_columns)
print("df.columns.values:", df_columns.values)

#%%
# columns = list(df.columns.values)   #기존 열 이름(List)
print("df.columns.values: ", type(df.columns.values)) # <class 'numpy.ndarray'>
columns = list(df.columns)   #기존 열 이름(Index)
print(columns, '\n')

#%%

# pclass의 0번째
print(df['pclass'][0]) # 3


#%%
# 열 이름을 알파벳 순으로 정렬하기 : 오름차순
columns_sorted = sorted(columns)    #알파벳 순으로 정렬
df_sorted = df[columns_sorted]
print(df_sorted, '\n')

#%%
# 열 이름을 기존 순서의 정반대 역순으로 정렬하기 : 내림차순
# columns_reversed = list(reversed(columns))  
# 오름차순으로 정렬된 것을 내림차순으로 정렬
columns_reversed = list(reversed(columns_sorted))  
df_reversed = df[columns_reversed]
print(df_reversed, '\n')

#%%
# 열 이름을 사용자가 정의한 임의의 순서로 재배치하기
columns_customed = ['pclass', 'sex', 'age', 'survived']  
df_customed = df[columns_customed]
print(df_customed)