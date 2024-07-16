# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:14:33 2024

@author: leehj
"""

# 라이브러리 불러오기
import pandas as pd

# 데이터셋 가져오기
data = pd.read_excel('./주가데이터.xlsx', engine= 'openpyxl')
print(data.head(), '\n')
print(data.dtypes, '\n')

#%%
data.to_csv("./주가데이터.csv", index=False)

#%%
df = pd.read_csv("./주가데이터.csv")

#%%
# 연, 월, 일 데이터 분리하기
df['연월일'] = df['연월일'].astype('str')   # 문자열 메소드 사용을 자료형 변경
dates = df['연월일'].str.split('-')        # 문자열을 split() 메서드로 분리
print(dates.head(), '\n') # 시리즈로 변환

#%%
# 분리된 정보를 각각 새로운 열에 담아서 df에 추가하기
# 문자열로 처리
df['연'] = dates.str.get(0)     # dates 변수의 원소 리스트의 0번째 인덱스 값
df['월'] = dates.str.get(1)     # dates 변수의 원소 리스트의 1번째 인덱스 값 
df['일'] = dates.str.get(2)     # dates 변수의 원소 리스트의 2번째 인덱스 값
print(df.head())

#%%

print("일자:", type(dates.iloc[0]), dates.iloc[0])
print("연:", dates.iloc[0][0])
print("월:", dates.iloc[0][1])
print("일:", dates.iloc[0][2])

#%%

for date in dates:
    print("일자:", ','.join(date))