# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:07:42 2024

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
    print(f"> addten: type:{type(n)}, len:{len(n)}") # <class 'pandas.core.series.Series'>
    print(n)
    print('-' * 30)
    return n + 10

#%%
"""
axis : {0 or 'index', 1 or 'columns'}, default 0
       Axis along which the function is applied:
   
       * 0 or 'index': apply function to each column.
       * 1 or 'columns': apply function to each row.
"""   

#%%

# 각 열 단위로 함수 적용
# 열 단위로 순회 : 컬럼 갯수 만큼 3번 순회
# 적용함수에는 한 컬럼 전체가 시리즈로 전달
# 한 컬럼에 해당하는 모든 891개 행
r1 = df.apply(addten, axis = 0)
print('=' * 30)
print(r1)

#%%