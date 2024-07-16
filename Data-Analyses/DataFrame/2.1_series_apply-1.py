# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:01:59 2024

@author: leehj
"""

"""
apply(func: 'AggFuncType', axis: 'Axis' = 0, raw: 'bool' = False, ...)

axis : {0 or 'index', 1 or 'columns'}, default 0
       Axis along which the function is applied:
    
       * 0 or 'index': apply function to each column.
       * 1 or 'columns': apply function to each row.
"""    

#%%
# 라이브러리 불러오기
import seaborn as sns

# titanic 데이터셋에서 age, fare 2개 열을 선택하여 데이터프레임 만들기
titanic = sns.load_dataset('titanic')
df = titanic.loc[:, ['age','fare']] # 컬럼: 나이, 요금
df['ten'] = 10   # 새로운 컬럼 생성
print(df.head())
print('\n')

#%%

# 사용자 함수 정의
def add_10(n):   # 10을 더하는 함수
    print(f"add_10({n})")
    return n + 10

def add_two_obj(n, b):    # 두 객체의 합
    return n + b

print(add_10(10)) # 20
print(add_two_obj(10, 10)) # 20
print('\n')

#%%

sr0 = df['age'] + 10

#%%

# 시리즈 객체에 적용
sr1 = df['age'].apply(add_10)   # n = df['age']의 모든 원소
print("sr1.shape:", sr1.shape, type(sr1))
print(sr1.head())
print('\n')

#%%  
# 시리즈 객체와 숫자에 적용 : 2개의 인수(시리즈 + 숫자)
sr2 = df['age'].apply(add_two_obj, b=10)    # n=df['age']의 모든 원소, b=10
print(sr2.head())
print('\n')

#%%

# 람다 함수 활용: 시리즈 객체에 적용
sr3 = df['age'].apply(lambda n: add_10(n))  # x=df['age']
print(sr3.head())

#%%

# 람다 함수 활용: 시리즈 객체에 적용
sr4 = df['age'].apply(lambda n: n + 10)  # x=df['age']
print(sr4.head())

#%%

help(df.apply)