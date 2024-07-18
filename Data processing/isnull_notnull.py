# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:59:36 2024

@author: leehj
"""

# -*- coding: utf-8 -*-

# 라이브러리 불러오기
import seaborn as sns

# titanic 데이터셋 가져오기
df = sns.load_dataset('titanic')

#%%

print(df.info())

#%%
# deck 열의 NaN 개수 계산하기
nan_deck = df['deck'].value_counts() 
print(nan_deck)


#%%

# 어떤 데크로 승선했는지 모르는 경우
# deck 열의 NaN 개수 계산하기 : dropnp=False
nan_deck = df['deck'].value_counts(dropna=False) # NaN : 688
print(nan_deck)

#%%

# null이면 True, 값이 있으면 False
# isnull() 메서드로 누락 데이터 찾기
print(df.head().isnull()) 

#%%
# 유효한 데이터는 True
# notnull() 메서드로 누락 데이터 찾기
print(df.head().notnull())

#%%

# 각 컬럼별로 누락 데이터 총갯수
# isnull() 메서드로 누락 데이터 개수 구하기
# axis:0
print(df.head().isnull().sum(axis=0))

#%%

# isnull() 메서드로 누락 데이터 개수 구하기
print(df.isnull().sum(axis=0)) # age:177, deck:688

#%%

# 각 행에서 누락된 데이터가 있는 컬럼의 갯수
print(df.isnull().sum(axis=1)) 