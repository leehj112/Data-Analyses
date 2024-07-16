# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:11:58 2024

@author: leehj
"""

# 라이브러리 불러오기
import seaborn as sns

# titanic 데이터셋에서 age, fare 2개 열을 선택하여 데이터프레임 만들기
titanic = sns.load_dataset('titanic')
df = titanic.loc[:, ['age','fare']]
print(df.head())
print('\n')

missing_value_cnt = 0

# 사용자 함수 정의
def missing_value(series):    # 시리즈를 인수로 전달
    global missing_value_cnt
    missing_value_cnt += 1
    return series.isnull()    # 불린 시리즈를 반환
    
# 데이터프레임의 각 열을 인수로 전달하면 데이터프레임을 반환
result = df.apply(missing_value, axis=0)  
print(result.head())
print('\n')
print(type(result))

# 2번 호출 : 각 칼럼 단위로 2번 호출
print("총 함수 호출 회수: ", missing_value_cnt) 

#%%

# 칼럼(age, fare)를 하나로 묶어서 True, False 갯수
result_counts = result.value_counts()
print(result_counts)

#%%

age_counts = result['age'].value_counts()
print(age_counts)

#%%

fare_counts = result['fare'].value_counts()
print(fare_counts)