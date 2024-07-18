# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:33:48 2024

@author: leehj
"""

# 라이브러리 불러오기
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# 정규화(normalization)
# 각 열에 속하는 데이터 값을 동일한 크기 기준으로 나눈 비율로 나타냄
# 값 : horsepower
# 범위: (값 - 최소값) / (최대값 - 최소값)
# 데이터의 범위: 0 ~ 1

# read_csv() 함수로 df 생성
df = pd.read_csv('./auto-mpg.csv', header=None)

# 열 이름을 지정
df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']  

# horsepower 열의 누락 데이터('?') 삭제하고 실수형으로 변환
df['horsepower'].replace('?', np.nan, inplace=True)      # '?'을 np.nan으로 변경
df.dropna(subset=['horsepower'], axis=0, inplace=True)   # 누락데이터 행을 삭제
df['horsepower'] = df['horsepower'].astype('float')      # 문자열을 실수형으로 변환

# horsepower 열의 통계 요약정보로 최대값(max)과 최소값(min)을 확인
print(df.horsepower.describe())
print('\n')

# horsepower 열의 최대값의 절대값으로 모든 데이터를 나눠서 저장
min_x = df.horsepower - df.horsepower.min()
min_max = df.horsepower.max() - df.horsepower.min()
df.horsepower = min_x / min_max

print(df.horsepower.head())
print('\n')
print(df.horsepower.describe())

#%%

# df['horsepower'] 데이터를 이용하여 차트를 그려라

# matplotlib 한글 폰트 오류 문제 해결
from matplotlib import font_manager, rc
font_path = "./malgun.ttf"   #폰트파일의 위치
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 스타일 서식 지정
plt.style.use('ggplot') 

# 그래프 객체 생성 (figure에 2개의 서브 플롯을 생성)
fig = plt.figure(figsize=(10, 10))   
ax1 = fig.add_subplot(2, 1, 1) # 행, 열, 순서
ax2 = fig.add_subplot(2, 1, 2) # 행, 열, 순서

# axe 객체에 plot 함수로 그래프 출력
ax1.plot(df['horsepower'], 'o', markersize=10)
ax2.plot(df['horsepower'], marker='o', markerfacecolor='green', markersize=10, 
         color='olive')
ax2.legend(loc='best')

#y축 범위 지정 (최소값, 최대값)
ax1.set_ylim(-1.0, 1.5)
ax2.set_ylim(-1.0, 1.5)

# 축 눈금 라벨 지정 및 75도 회전
ax1.set_xticklabels(df['mpg'])
ax2.set_xticklabels(df['mpg'])

plt.show()  # 변경사항 저장하고 그래프 출력