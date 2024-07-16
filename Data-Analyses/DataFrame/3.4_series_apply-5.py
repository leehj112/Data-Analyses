# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:06:37 2024

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
# axis = 1 : 각 행단위로 인자가 전달
# n : Series
def addten(n):
    print(f"> addten: {type(n)}, ({n})")
    return n['age'] + n['ten']

#%%

# [문제] 칼럼: age, ten을 apply() 함수로 전달하여 
# age + ten의 결과를 새로운 칼럼 ageten 만들어서 넣어라.
# 람다 함수 활용: 시리즈 객체에 적용

#%%

"""
axis : {0 or 'index', 1 or 'columns'}, default 0
       Axis along which the function is applied:
   
       * 0 or 'index': apply function to each column.
       * 1 or 'columns': apply function to each row.
"""   

#%%

# 각 행 단위로 함수 적용
# 적용함수에는 한 행 전체가 시리즈로 전달
ageten2 = df.apply(addten, axis = 1)

#%%

ageten = df.apply((lambda n : n['age'] + n['ten']), axis = 1)
df['ageten'] = ageten
print(df.head())