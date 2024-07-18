# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:11:55 2024

@author: leehj
"""

# -*- coding: utf-8 -*-

# 라이브러리 불러오기
import pandas as pd

# read_csv() 함수로 df 생성
df = pd.read_csv('./auto-mpg.csv', header=None)

# 열 이름을 지정
df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name'] 
print(df.head(3))    
print('\n')

#%%
"""
1갤런(3.78541 리터)
1마일(1.6093 Km)
1리터 당 연비 : 1.6093 / 3.78541 -> 0.425Km/L
18mpg(마일) : 1갤런 연비 -> 18 * 0.425 -> 7.65Km/L
 
18 * 1.6Km = 28.8Km / 3.78541 = 7.6Km/L
"""
#%%
# mpg(mile per gallon)를 kpl(kilometer per liter)로 변환 (mpg_to_kpl = 0.425)
mpg_to_kpl = 1.60934 / 3.78541 # 0.425

# mpg 열에 0.425를 곱한 결과를 새로운 열(kpl)에 추가
df['kpl'] = df['mpg'] * mpg_to_kpl
print(df['kpl'].head(3))    
print('\n')

#%%
# kpl 열을 소수점 아래 둘째 자리에서 반올림 
df['kpl'] = df['kpl'].round(2)
print(df['kpl'].head(3))     