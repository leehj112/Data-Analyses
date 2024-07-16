# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:12:43 2024

@author: leehj
"""

# 라이브러리 불러오기
import seaborn as sns

# titanic 데이터셋에서 age, fare 2개 열을 선택하여 데이터프레임 만들기
titanic = sns.load_dataset('titanic')
df = titanic.loc[:, ['age','fare']]
print(df.head())
print('\n')

#%%
# 사용자 함수 정의
def min_max(x):    # 최대값 - 최소값
    return x.max() - x.min()
    
# 데이터프레임의 각 열을 인수로 전달하면 시리즈를 반환
result = df.apply(min_max)   # 기본값 axis=0 
print(result)
print('\n')
print(type(result)) # age(79.58), fare(512.329)
# <class 'pandas.core.series.Series'>
for idx in result.index:
    print(idx, ":", result[idx])

#%%

# [검산]
print("# [검산]")

age = df['age'].max() - df['age'].min()
fare = df['fare'].max() - df['fare'].min()

print(f"age({age}), fare({fare})") # age(79.58), fare(512.3292)