# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:12:56 2024

@author: leehj
"""

# 데이터프레임 객체를 함수에 매핑

# 라이브러리 불러오기
import seaborn as sns

# titanic 데이터셋에서 age, fare 2개 열을 선택하여 데이터프레임 만들기
titanic = sns.load_dataset('titanic')
df = titanic.loc[:, ['age','fare']]

#%%
# 각 열의 NaN 찾기 - 데이터프레임 전달하면 데이터프레임을 반환
def missing_value(x):    
    print("[missing_value]")
    print(x)
    print("-" * 50)
    return x.isnull()    

# 각 열의 NaN 개수 반환 - 데이터프레임 전달하면 시리즈 반환
def missing_count(x):    # 
    return missing_value(x).sum()

# 데이터프레임의 총 NaN 개수 - 데이터프레임 전달하면 값을 반환
def totoal_number_missing(x):    
    return missing_count(x).sum()
    
#%%

# age, fare에서 NaN은 True로 세팅
# 데이터프레임에 pipe() 메소드로 함수 매핑
result_df = df.pipe(missing_value)   

print(result_df.head())
print(type(result_df))
print('\n')

#%%

# 각가 age, fare에서 NaN의 갯수
result_series = df.pipe(missing_count)   
print(result_series)
print(type(result_series))
print('\n')

#%%

# age, fare에서 NaN의 총합
result_value = df.pipe(totoal_number_missing)   
print(result_value)
print(type(result_value))