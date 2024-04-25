# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:23:00 2024

@author: leehj
"""

# 막대그래프 
import pandas as pd

df = pd.read_excel('./data/part3/남북한발전전력량.xlsx', engine='openpyxl')

df_ns = df.iloc[[0,5],3:]
df_ns.index = ['South','North']
df_columns = df_ns.columns.map(int)  # --> 자료형을 정수형 

# 행열 전치 
tdf_ns = df_ns.T
print(tdf_ns.head())
print('\n')
tdf_ns.plot(kind='bar') 

#%% 
# 히스토그램 
import pandas as pd

df = pd.read_excel('./data/part3/남북한발전전력량.xlsx', engine='openpxl')

df_ns = df.iloc[[0,5],3:]
df_ns.index = ['South','North'] 
df_ns.columns = df_ns.columns.map(int)

tdf_ns = df_na.T
tdf_ns.plot(kind='hiat') 

#%% 
# 산점도 
import pandas as pd

df = pd.read_csv('./data/part3/auto-mpg.csv', header=None)

df.columns = ['mpg', 'cyilnders', 'displacement', 'horsepower', 'weight',
              'acceleration','model year', 'origin', 'name']

df.plot(x='weight', y='mpg', kind='scatter') 

#%%
# 박스 플롯
import pandas as pd

df = pd.read_csv('./data/part3/auto-mpg.csv',header=None) 

df.columns = ['mpg','cylinders', 'displacement', 'horsepower', 'weight',
              'acceleration','model year', 'origin', 'name']

df[['mpg','cylinders']].plot(kind='box')

#%% 
# 선 그래프

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('data/part4/시도별 전출입 인구수.xlsx', engine = 'openpyxl', header=0)


df = df.fillna(method='ffill')

mask = (df['전출지별'] ==  '서울특별시') & (df['전입지별'] != '서울특별시') 
df_seoul = df[mask]
df_seoul = df_seoul.drop(['전출지별'], axis=1) 
df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace=True)
df_seoul.set_index('전입지', inplace=True) 


#서울에서 경기도 이동안 인구 데이터 삭제
sr_one = df_seoul.loc['경기도']
#%%
# x, y축 데이터를 plot 함수에 입력 
plt.plot(sr_one.index, sr_one. values) 
plt.plot(sr_one) 


#%%
# 차트 제목, 축 이름 추가
# 서울에서 경기도로 이동한 인구 데이터 값만 선택 
sr_one = df_seoul.loc['경기도']

# x, y축 데이터를 plot 함수에 입력
plt.plot(sr_one.index, sr_one.values)

# 차트 제목 추가
plt.title('서울 -> 경기 인구 이동') 

plt.xlabel('기간')
plt.ylabel('이동 인구수')

plt.show 

#%% 
# Matplotlib 한글 폰트 오류 해결 

from matplotlib import font_manager, rc
font_path = "./data/part4/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name) 

#%%
import pandas as pd
import matplotlib.pyplot as plt 

from matplotlib import font_manager, rc
font_path = "./data/part4/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name) 


# Excel 데이터를 데이터프레임으로 변환 
df = pd.read_excel('./data/part4/시도별 전출입 인구수.xlsx', engine='openpyxl', header=0) 

df = df.fillna(method='ffill')

mask = (df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시')
df_seoul = df[mask]
df_seoul = df_seoul.drop(['전출지별'], axis=1)
df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace=True)
df_seoul.set_index('전입지', inplace=True) 



# 서울에서 경기도로 이동한 인구 데이터 값만 선택
sr_one = df_seoul.loc['경기도']

# x,y축 데이터를 plot함수에 입력
plt.plot(sr_one.index, sr_one.values) 


# 차트 제목 추가
plt.title('서울 => 경기 인구 이동')

plt.xlabel('기간')
plt.ylabel('이동 인구수')

plt.show () 



#%%
# 서울에서 경기도로 이동한 인구 데이터 값만 선택  
sr_one = df_seoul.loc['경기도']

plt.figure(figsize=(14,5)) 

#%%
# x축 눈금 라벨 회전
plt.xticks(rotation = 'vertical') 

plt.plot(sr_one.index, sr_one.values)

plt.title('서울 -> 경기 인구 이동')
plt.xlabel('기간')
plt.ylabel('이동 인구수')

plt.legend(labels=['서울 -> 경기'], loc='beat') # 범례 
plt.show() 

