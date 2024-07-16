# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:04:04 2024

@author: leehj
"""

# 라이브러리 불러오기
import seaborn as sns
import pandas as pd

# titanic 데이터셋
titanic = sns.load_dataset('titanic')
titanic.info()

#%%

# [문제]
# 고객의 나이에 따른 요금 차등 계산
# 20세 이하: 70%
# 50세 이하: 100%
# 51세 이상: 50%
    
#%%
"""
axis : {0 or 'index', 1 or 'columns'}, default 0
       Axis along which the function is applied:
   
       * 0 or 'index': apply function to each column.
       * 1 or 'columns': apply function to each row.
"""   
#%%

def user_fare_apply(df, age, fare, newcol, func):
    cols = [age, fare]
    ndf = df.loc[:, cols] # 새로운 데이터프레임
    ndf[newcol] = ndf.apply(func, axis=1) # 새로운 요금 컬럼
    return ndf

#%%

# axis=1: apply function to each row.
# 행단위로 데이터를 시리즈로 받음
def user_fare(ncols):
    age = ncols.loc['age']
    fare = ncols.loc['fare']
    if age <= 20:
        return fare * 0.7
    elif age <= 50:
        return fare
    else:
        return fare * 0.5

#%%

# 연령별로 요금을 차등
ndf = user_fare_apply(titanic.iloc[0:20,:], 'age', 'fare', 'tot', user_fare)
print(ndf)