# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:56:05 2024

@author: leehj
"""

# 라이브러리 불러오기
import pandas as pd
import seaborn as sns

# titanic 데이터셋에서 age, sex 등 5개 열을 선택하여 데이터프레임 만들기
titanic = sns.load_dataset('titanic')
df = titanic.loc[:, ['age','sex', 'class', 'fare', 'survived']]

# class 열, sex 열을 기준으로 분할
grouped = df.groupby(['class', 'sex'])  

# 그룹 객체에 연산 메서드 적용
gdf = grouped.mean()
print(gdf)
print('\n')
print(type(gdf))

# class 값이 First인 행을 선택하여 출력
print(gdf.loc['First'])
print('\n')

# class 값이 First이고, sex 값이 female인 행을 선택하여 출력
print(gdf.loc[('First', 'female')])
print('\n')

# sex 값이 male인 행을 선택하여 출력
print(gdf.xs('male', level='sex'))