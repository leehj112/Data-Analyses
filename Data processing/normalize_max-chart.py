# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:32:48 2024

@author: leehj
"""

# -*- coding: utf-8 -*-

# 정규화(normalization)
# 각 열에 속하는 데이터 값을 동일한 크기 기준으로 나눈 비율로 나타냄
# 데이터의 범위: 0 ~ 1, -1 ~ 1
# 최대값 : 1
# horsepower 열의 최대값의 절대값으로 모든 데이터를 나눠서 저장

# 라이브러리 불러오기
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# read_csv() 함수로 df 생성
df = pd.read_csv('./auto-mpg.csv', header=None)

# 열 이름을 지정
df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']  

# horsepower 열의 누락 데이터('?') 삭제하고 실수형으로 변환
df['horsepower'].replace('?', np.nan, inplace=True)      # '?'을 np.nan으로 변경
df.dropna(subset=['horsepower'], axis=0, inplace=True)   # 누락데이터 행을 삭제
df['horsepower'] = df['horsepower'].astype('float')      # 문자열을 실수형으로 변환

# horsepower 열의 통계 요약정보로 최대값(max)을 확인
print(df.horsepower.describe())
print('\n')

#%%

# 최대값 : 1
# horsepower 열의 최대값의 절대값으로 모든 데이터를 나눠서 저장
df.horsepower = df.horsepower / abs(df.horsepower.max()) 

print(df.horsepower.head())
print('\n')
print(df.horsepower.describe())


#%%

# df['horsepower'] 데이터를 이용하여 차트를 그려라
# 스타일 테마 설정 (5가지: darkgrid, whitegrid, dark, white, ticks)
sns.set_style('darkgrid')

# 그래프 객체 생성 (figure에 2개의 서브 플롯을 생성)
fig = plt.figure(figsize=(15, 5))   
ax = fig.add_subplot(1, 1, 1)
 
# 그래프 그리기 - 선형회귀선 표시(fit_reg=True)
sns.regplot(x='mpg',        #x축 변수
            y='horsepower',       #y축 변수
            data=df,   #데이터
            ax=ax)         #axe 객체 - 1번째 그래프 

plt.show()