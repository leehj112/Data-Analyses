# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:03:19 2024

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
# 위의 데이터셋으로 다음의 처리를 하는 함수를 정의하라.
# 함수이름: user_dataframe_apply
# 파라미터: 데이터프레임, 컬럼목록, 새로운컬럼, 처리함수, 전달인자
# 기능설명:
#   - 데이터프레임: 처리대상 데이터프레임
#   - 컬럼목록: 처리대상 컬럼목록
#   - 새로운컬럼: 처리대상 컬럼목록으로 처리함수를 적용한 결과 컬럼
#   - 처리함수: 처리해야할 기능 함수
#   - 전달인자: 처리함수에 전달할 인자
# 처리함수: 처리로직은 선택
# 리턴: 새로운 데이터프레임
    
#%%

def user_dataframe_apply(df, cols, newcol, func, args):
    ndf = df.loc[:, cols] # 새로운 데이터프레임
    ndf[newcol] = 0.0     # 새로운 컬럼럼
    
    for n in range(len(ndf)):
        lx = ndf.index[n]
        ncols = ndf.loc[lx, cols].astype(float)
        ndf.loc[lx, newcol] = func(ncols, args)
        
    return ndf

#%%

def user_func(ncols, args):
    tot = 0.0
    for col in ncols:
        if pd.isnull(col) != True and pd.isna(col) != True:
            tot += col
        
    return tot + args

#%%

def user_mean(ncols, args):
    tot = 0.0
    for col in ncols:
        if pd.isnull(col) != True and pd.isna(col) != True:
            tot += col
        
    return tot / args


#%%

# 대상의 모든 컬럼을 더하고 가중치를 더함
ndf1 = user_dataframe_apply(titanic.iloc[0:10,:], ['age', 'fare'], 'weight', user_func, 10.0)
print(ndf1)

#%%

# 대상의 모든 컬럼으 평균
ndf2 = user_dataframe_apply(titanic.iloc[0:10,:], ['age', 'fare'], 'avg', user_mean, 2.0)
print(ndf2)