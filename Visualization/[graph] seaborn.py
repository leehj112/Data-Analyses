# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:41:34 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# 시본 데이터 시각화
# https://seaborn.pydata.org/api.html

# pip install seaborn

#%%


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib 
import matplotlib.pyplot as plt

#%%

print(sns.__version__) # 0.13.2

#%%

# 테마지정
sns.set_style("whitegrid")

#%%

# 비행기 데이터셋
flights = sns.load_dataset('flights')
flights.head()

#%%

###############################################################################
# 수치형 데이터
###############################################################################

#%%

# 라인그래프
# x: 연도
# y: 승객수
# 라인 주위의 투명하고 두꺼운 면적은 연도마다 반복되는 값이 집계되어 
# 평균과 95% 신뢰구간을 나타님
sns.lineplot(x='year', y='passengers', data=flights)

#%%

# query() 함수 사용
# 9월 데이터만 한정해서 시각화
september_flights = flights.query("month == 'Sep'")
sns.lineplot(x='year', y='passengers', data=september_flights)

#%%

# hue: 구분하고자 하는 변수 지정
# 전체 평균값을 1개로 표현한 라인을 월 기준으로 분리
sns.lineplot(x='year', y='passengers', hue='month', data=flights)

#%%

###############################################################################
# 데이터셋: tips
# 레스토랑에 방문한 손님이 팁을 얼마나 주는지, 
# 성별, 흡연여부, 요일, 식사 시간, 식사 인원 등에 대한 정보를 가지고 있는 데이터 셋
"""
컬럼명      의미        인자             자료형(data type)
total_bill  요금(달러)  3.07~50.81       실수(float)
tip팁       달러        1.0~10.0         실수(float)
sex         성별        Male/Female      문자열(str)
smoker      흡연여부    Yes/No           문자열(str)
day         요일        Thu,Fri,Sat,Sun  문자열(str)
time        식사시간    Lunch, Dinner    문자열(str)
size        식사인원    1~6              정수(int)
"""
###############################################################################

tips = sns.load_dataset('tips')
tips.head()

#%%
"""
   total_bill   tip     sex smoker  day    time  size
0       16.99  1.01  Female     No  Sun  Dinner     2
1       10.34  1.66    Male     No  Sun  Dinner     3
2       21.01  3.50    Male     No  Sun  Dinner     3
3       23.68  3.31    Male     No  Sun  Dinner     2
4       24.59  3.61  Female     No  Sun  Dinner     4
"""

# In[63]:

# 라인그래프를 동시에 2개 그리기: 비교
# sns.relplot(x='day', y='tip', kind='line', col='sex', hue='smoker', ci=None, data=tips)
sns.relplot(x='day', y='tip', kind='line', col='sex', hue='smoker', errorbar=None, data=tips)

#%%

###############################################################################
# 데이터분포 확인
###############################################################################

penguins = sns.load_dataset('penguins')
penguins.head()


#%%

# 히스토그램
# 기본 : 10개 막대그래프
# flipper_length_mm: 지느러미 길이
sns.histplot(x='flipper_length_mm', data=penguins)


#%%

# 막대그래프 수: bins=20, 20개
sns.histplot(x='flipper_length_mm', bins=20, data=penguins)


#%%

# 막대그래프의 갯수가 아니라 너비를 조정
# binwidth: 6(기본), 숫자가 작을수록 더 세밀한 막대 그래프로 표현
sns.histplot(x='flipper_length_mm', binwidth=2, data=penguins)


# In[54]:

# 수평 막대그래프
# y 데이터를 지정
sns.histplot(y='flipper_length_mm', data=penguins)

#%%

# hue : 그룹별로 세분화
# species: 펭귄 종류별 분포
sns.histplot(x='flipper_length_mm', hue='species', data=penguins)

#%%

# multiple: 'stack'
# 각 구간에 종별 막대그래프가 쌓인 형태
sns.histplot(x='flipper_length_mm', hue='species', multiple='stack', data=penguins)


#%%

# 다중 히스토그램: displot()
# col : 컬럼
sns.displot(x='body_mass_g', hue='species', 
            col='sex',  # 분리 기준 변수
            kind='hist', 
            multiple='stack', data=penguins)

#%%

###############################################################################
# 수치형 데이터 상관관계
###############################################################################
# 데이터셋: tips
###############################################################################

tips = sns.load_dataset('tips')
tips.head()

#%%

# 산점도
# 레스토랑에서 손님이 지불한 총액과 팁
# 상관관계 : 양의 상관관계로 지불액이 커질수록 팁도 커짐
sns.scatterplot(x='total_bill', y='tip', data=tips)

#%%

# 시간대를 분리
# hue: 'time', 점심, 저녁
sns.scatterplot(x='total_bill', y='tip', hue='time', data=tips)


#%%

# 요일별로 분리
# hue: 'day'
sns.scatterplot(x='total_bill', y='tip', hue='day', style='time', data=tips)


#%%

# 데이터의 수치에 따라 산점도 사이즈 변경
# size: 'size', 식사인원
# sizes: (20, 200), 도형의 최소, 최대 크기 지정
sns.scatterplot(x='total_bill', y='tip', 
                hue='size', 
                size='size', 
                sizes=(20, 200), 
                legend='full', data=tips)


#%%

# relplot()
# 산점도를 여러개 표시 : displot()과 유사
sns.relplot(x='total_bill', y='tip', 
            col='time', 
            hue='day', 
            style='day', 
            kind='scatter', data=tips)

#%%


sns.stripplot(x='day', y='total_bill', data=tips)


#%%

#sns.stripplot(x='day', y='total_bill', data=tips, jitter=0.05, marker='D', size=20, edgecolor="gray", alpha=.25)
sns.stripplot(x='day', y='total_bill', data=tips, jitter=0.05, marker='D', size=20, alpha=.25)


#%%

sns.stripplot(x='day', y='total_bill', hue='smoker', palette='Set2', dodge=True, data=tips)


#%%
###############################################################################
# 범주형 데이터 갯수확인
###############################################################################

tips = sns.load_dataset('tips')
tips.head()

#%%

# 막대그래프
# 관측값의 갯수
sns.countplot(x='day', data=tips)

#%%

# 그룹별로 구분
# hue: 'time' 시간대별
sns.countplot(x='day', hue='time', data=tips)

#%%

# 한 번에 여러개의 countplot()을 시각화
# kind: 'count'
sns.catplot(x='sex', hue='time', 
            col='day',
            data=tips, 
            kind='count',
            height=5, aspect=.6)

#%%

###############################################################################
# 범주형과 수치형 데이터
###############################################################################

# 막대그래프
# 각 막대의 높이: 입력한 수치형 변수에 대한 중심 경향의 추정치
# 막대 상단의 오차막대: 행당 추정치 주변의 불활실성
# x : 'day' 요일
# y : 'total_bill' 지불액
sns.barplot(x='day', y='total_bill', data=tips)


#%%

# 성별로 구분
sns.barplot(x='day', y='total_bill', hue='sex', data=tips)


#%%

# 오차막대 신뢰구간의 수치와 색상 팔레트 속성 지정
# sns.barplot(x='day', y='total_bill', hue='sex', data=tips, ci=65, palette='Blues_d')
sns.barplot(x='day', y='total_bill', hue='sex', data=tips, errorbar=('ci', 65), palette='Blues_d')


#%%

# palette: https://seaborn.pydata.org/tutorial/color_palettes.html
# 동시에 여러개의 막대 그래프 표시
# kind: 'bar' barplot(막대)
# col: 'time' 시간대별로 구분(점심, 저녁)
sns.catplot(x='day', y='total_bill', hue='smoker', col='time',
            data=tips, kind='bar', palette='Blues_d', height=5, aspect=.9);


#%%

###############################################################################
# 범주형과 수치형 데이터
###############################################################################

# 박스플롯
sns.boxplot(x='total_bill', data=tips)


#%%

sns.boxplot(x='day', y='total_bill', data=tips)


#%%

# hue: 'sex'
# 그룹별 : 성별
sns.boxplot(x='day', y='total_bill', 
            hue='sex', 
            data=tips, linewidth=2.5, 
            palette='Set2')


#%%

# 4분위수 기준의 데이터 분포와 이상값 확인
# 박스플롯과 산점도 결합
# stripplot : 하나의 변수에 대한 산점도

# sns.boxplot(x='day', y='total_bill', data=tips, palette='Set2')
sns.boxplot(x='day', y='total_bill', data=tips, hue='day', palette='Set1', legend=False)

sns.stripplot(x='day', y='total_bill', data=tips, color=".25")


#%%

# swarmplot : 산점도의 점들을 겹치지 않게 
# sns.boxplot(x='day', y='total_bill', data=tips, palette='Set2')
sns.boxplot(x='day', y='total_bill', data=tips, hue='day', palette='Set1', legend=False)
sns.swarmplot(x='day', y='total_bill', data=tips, color=".25")


#%%

###############################################################################
# 범주형과 수치형 데이터
###############################################################################

#%%

# 바이올린 그래프
# sns.violinplot(x='day', y='total_bill', hue='sex', data=tips, palette='Set1', scale='count')
sns.violinplot(x='day', y='total_bill', hue='sex', data=tips, palette='Set1', density_norm='count')


#%%

# hue : 데이터의 고윳값이 개별 바이올린으로 분리
# split : hue를 기준으로 분리된 바이올린을 하나의 바이올린으로 출력
# inner : 4분위로 동시 표시
sns.violinplot(x='day', y='total_bill', hue='sex', 
               data=tips, 
               palette='Set1', 
               split=True, 
               density_norm='count', 
               inner='quartile')


#%%

# 기본 바이올린플롯에서 좀 더 분포의 모양을 반영한 모습
# bw_method : bandwidth(대역폭)을 줄임

# sns.violinplot(x='day', y='total_bill', hue='sex', data=tips, palette='Set1', split=True, density_norm='count', inner='stick', scale_hue=False, bw=.2)

sns.violinplot(x='day', y='total_bill', hue='sex', data=tips, palette='Set1', split=True, 
               density_norm='count', inner='stick',
               scale_hue=False, bw_method=0.2)


#%%

###############################################################################

planets = sns.load_dataset('planets')
planets.head()


#%%

# 가로형: y에 범주형 변수를 지정 

"""
sns.violinplot(x='orbital_period', y='method',
                    data=planets[planets.orbital_period < 1000],
                    scale='width', palette='Set2')
"""

sns.violinplot(x='orbital_period', 
               y='method', 
               data=planets[planets.orbital_period < 1000], 
               hue='method', 
               density_norm='width', 
               palette='Set2', 
               legend=False)


#%%

# 데이터의 시작부분을 자름
# cut : 0, 마이너스 자름

"""
sns.violinplot(x='orbital_period', y='method',
                    data=planets[planets.orbital_period < 1500],
                    cut=0, scale='width', palette='Set2')
"""

sns.violinplot(x='orbital_period', 
               y='method', 
               data=planets[planets.orbital_period < 1500], 
               cut=0,
               hue='method', 
               density_norm='width', 
               palette='Set2', 
               legend=False)


#%%
# THE END



