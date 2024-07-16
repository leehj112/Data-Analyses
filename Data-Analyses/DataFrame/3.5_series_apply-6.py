# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:07:15 2024

@author: leehj
"""

# 라이브러리 불러오기
import seaborn as sns

# titanic 데이터셋에서 age, fare 2개 열을 선택하여 데이터프레임 만들기
titanic = sns.load_dataset('titanic')
df = titanic.loc[:, ['age','fare']]
df['ten'] = 10
print(df.head())
print('\n')

#%%

# 사용자 함수 정의
def addten(n):
    print(f"> addten({n})")
    return n['age'] + n['ten']

#%%

# [문제] 칼럼: age, ten을 apply() 함수로 전달하여 
# age + ten의 결과를 새로운 칼럼 ageten 만들어서 넣어라.
# 람다 함수 활용: 시리즈 객체에 적용

#%%

# 각 행에 대한 칼럼 단위로 연산을 수행
# 각 행에 대해 사용자 함수 1번 호출
# 행의 수 만큼 사용자 함수 호출
ageten = df.apply(addten, axis = 1)

#%%
df['ageten'] = ageten
print(df.head())