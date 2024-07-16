# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:49:58 2024

@author: leehj
"""

# 라이브러리 불러오기
import seaborn as sns
import pandas as pd

# titanic 데이터셋 로딩
titanic = sns.load_dataset('titanic')

# IPyhton 디스플레이 설정 변경 - 출력할 최대 열의 개수
pd.set_option('display.max_columns', 10)  
    
# 함께 탑승한 형제 또는 배우자의 수가 3, 4, 5인 승객만 따로 추출 - 불린 인덱싱
mask3 = titanic['sibsp'] == 3
mask4 = titanic['sibsp'] == 4
mask5 = titanic['sibsp'] == 5
df_boolean = titanic[mask3 | mask4 | mask5]
print(df_boolean.head())
print('\n')

#%%

# isin() 메서드 활용하여 동일한 조건으로 추출
# 'sibsp' 칼럼의 값이 [3,4,5] 중에 하나이면 선택
isin_filter = titanic['sibsp'].isin([3, 4, 5])
df_isin = titanic[isin_filter]
print(df_isin.head())