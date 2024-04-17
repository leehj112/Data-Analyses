# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:21:48 2024

@author: leehj
"""


# 맵플롭립(matplotlib)
# 색상 이름: https://matplotlib.org/stable/gallery/color/named_colors.html
# HEX 코드: https://htmlcolorcodes.com/
# 라인스타일: https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
# 컬럼맵: https://matplotlib.org/stable/tutorials/colors/colormaps.html

#%%

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib 
import matplotlib.pyplot as plt

#%%

print(matplotlib.__version__) # 3.8.4
print(sns.__version__)        # 0.13.2

#%%

plt.plot([2, 1, 5, 3, 7, 8, 13])
plt.show()


#%%

# 넘파이 배열: 0부터 9까지 숫자
data = np.arange(10)
print(data) # [0 1 2 3 4 5 6 7 8 9]

plt.plot(data)
plt.show()

#%%

# x, y값 동시
dx = [1, 2, 3, 4]
dy = [2, 3, 5, 10]

plt.plot(dx, dy)
plt.show()


#%%

# 데이터프레임을 이용하여 그래프 출력

import seaborn as sns
iris = sns.load_dataset('iris')


#%%

# 데이터프레임을 컬럼으로 사용하여 출력
plt.plot(iris.sepal_length)
plt.show()


#%%

# x, y축 레이블 설정
dx = [1, 2, 3, 4]
dy = [1, 7, 5, 12]

plt.plot(dx, dy)
plt.xlabel('X Axis Label', labelpad=12, fontdict={'color': 'hotpink', 'size': 14})    #  폰트타입 설정 >> 'family' : 'arial'
plt.ylabel('Y Axis Label', labelpad=12, fontdict={'color': 'k', 'size': 14})          #  폰트두께 설정 >> 'weight' : 'normal'  
plt.show()


#%%

# 범례표시: legend()
plt.plot(dx, dy, label='Data A')
plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')
plt.legend()       # 위치 숫자로 입력 가능 legend(loc=(1.0, 1.0))
plt.show()

#%%

# 2개 이상의 데이터가 동시에 그려지는 경우
# 범례의 열 개수 지정:  기본1, ncol=2
plt.plot(dx, dy, label='Data A')
plt.plot(dy, dx, label='Data B')
plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')
plt.legend(loc='best', ncol=2)  
plt.show()

#%%
# loc 입력 가능 옵션 :
# 
# best,
# upper right,
# upper left,
# lower left,
# lower right,
# right,
# center left,
# center right,
# lower center,
# upper center,
# center

#%%

data_x = [1, 2, 3, 4 ,5]
data_y = [2, 5, 3, 6, 9]

#%%

# 레이블 설정 및 범위 지정
plt.plot(data_x, data_y)
plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')
plt.xlim([0, 5])   
plt.ylim([-2, 20])    
plt.show()


#%%

# x, y축 범위 지정    
# axis : [x축 최솟값, x축 최댓값, y축 최솟값, x축 최댓값]
plt.plot(data_x, data_y)
plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')
plt.axis([0, 5, -2, 20]) 
plt.show()

#%%

# 컬러 지정
# 
plt.plot(data_x, data_y, color='g')     # color='#ff0000' 형식도 가능
plt.show()


#%%

# 라인스타일
plt.plot([1, 2, 3, 4], [2, 1, 3, 5], linestyle='solid', color='g', label='solid')
plt.plot([1, 2, 3, 4], [3, 2, 4, 6], linestyle='dashed', color='g', label='dashed')
plt.plot([1, 2, 3, 4], [4, 3, 5, 7], linestyle='dotted', color='g', label='dotted')
plt.plot([1, 2, 3, 4], [5, 4, 6, 8], linestyle='dashdot', color='g', label='dashdot')

plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')
plt.legend(loc='best')

plt.axis([0.5, 4.5, 0, 9])
plt.show()


#%%

# 라인두께
plt.plot([1, 2, 3, 4], [2, 1, 3, 5], linestyle='solid', solid_capstyle='butt', color='g', linewidth=8, label='solid')
plt.plot([1, 2, 3, 4], [3, 2, 4, 6], linestyle='solid', solid_capstyle='round', color='g', linewidth=8, label='dashed')
plt.plot([1, 2, 3, 4], [4, 3, 5, 7], linestyle='dashed', dash_capstyle='butt', color='lightgreen', linewidth=8, label='dotted')
plt.plot([1, 2, 3, 4], [5, 4, 6, 8], linestyle='dashed', dash_capstyle='round', color='lightgreen',linewidth=8, label='dashdot')

plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')
plt.legend(loc='best')

plt.axis([0.5, 4.5, 0, 9])
plt.show()


#%%

# 마커
plt.plot([1, 2, 3, 4], [2, 1, 3, 5], marker='o', color='g', label='data A marker')
plt.plot([1, 2, 3, 4], [4, 3, 5, 7], marker='s', color='limegreen', label='data B marker')

plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')
plt.legend(loc='best')

plt.axis([0.5, 4.5, 0, 8])
plt.show()


#%%

# 마커
plt.plot([1, 2, 3, 4], [2, 1, 3, 5], 'o--', color='g', label='solid + marker')
plt.plot([1, 2, 3, 4], [4, 3, 5, 7], 'x:', color='limegreen', label='dashed + marker')

plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')
plt.legend(loc='best')

plt.axis([0.5, 4.5, 0, 8])
plt.show()

#%%

# 그리드: grid
plt.plot([1, 2, 3, 4], [2, 1, 3, 5], linestyle='solid', color='g', label='solid')
plt.plot([1, 2, 3, 4], [4, 3, 5, 7], 'x:', color='limegreen', label='dashed + marker')

plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')
plt.legend(loc='best')
# plt.grid(False)
plt.grid(True)
plt.show()


#%%

# 그리드: grid
# 그리드가 표시되는 방향
# 컬러, 투명도, 라인스타일
plt.plot([1, 2, 3, 4], [2, 1, 3, 5], linestyle='solid', color='g', label='solid')
plt.plot([1, 2, 3, 4], [4, 3, 5, 7], 'x:', color='limegreen', label='dashed + marker')

plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')
plt.legend(loc='best')
plt.grid(True, axis='y', color='orange', alpha=0.25, linestyle='-.')
plt.show()

#%%

# 타이틀(title)
# loc : 위치
# pad : 그래프와의 공백
plt.plot([1, 2, 3, 4], [2, 1, 3, 5], linestyle='solid', color='g', label='solid')

plt.legend(loc='best')
plt.grid(True, axis='y', alpha=0.25)

font_style = {
    'fontsize': 14,
    'fontweight': 'bold'   #  {‘normal’,‘bold’, 'heavy’,‘light’,‘ultrabold’,‘ultralight’} 설정 가능
}

plt.title('Title of Chart', fontdict=font_style, loc='center', pad=15)
plt.show()

#%%

# 여러 그래프 동시 그리기
# subplot(행, 열, 인덱스)

# 레이아웃: 2행 1열
plt.subplot(2, 1, 1)  # 첫 번째
plt.plot([1, 2, 3, 4], [3, 4, 1, 7], linestyle='solid', color='g', label='solid')
plt.xlabel('speed')
plt.ylabel('Data A')

plt.subplot(2, 1, 2) # 두 번째
plt.plot([1, 2, 3, 4], [5, 2, 9, 6], linestyle='solid', color='limegreen', label='solid')
plt.xlabel('speed')
plt.ylabel('Data B')

plt.tight_layout()
plt.show()

#%%

ax1 = plt.subplot(2, 1, 1)
plt.plot([1, 2, 3, 4], [3, 4, 1, 7], linestyle='solid', color='g', label='solid')
plt.ylabel('Data A')
plt.xticks(visible=False)

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot([1, 2, 3, 4], [5, 2, 9, 6], linestyle='solid', color='limegreen', label='solid')
plt.xlabel('speed')
plt.ylabel('Data B')

plt.tight_layout()
plt.show()


# In[19]:

# 레이아웃: 1행 2열

plt.subplot(1, 2, 1) # 첫 번째
plt.plot([1, 2, 3, 4], [3, 4, 1, 7], linestyle='solid', color='g', label='solid')
plt.xlabel('speed')
plt.ylabel('Data A')

plt.subplot(1, 2, 2) # 두 번째
plt.plot([1, 2, 3, 4], [5, 2, 9, 6], linestyle='solid', color='limegreen', label='solid')
plt.xlabel('speed')
plt.ylabel('Data B')

plt.tight_layout()
plt.show()


#%%
###############################################################################
# 데이터셋: iris를 이용한 예제
###############################################################################

iris = sns.load_dataset('iris')
iris.head()

#%%

# 히스토그램
plt.hist(iris.sepal_length, color='skyblue')
plt.show()

#%%

# 2개 이상의 변수 히스토그램
plt.hist(iris.sepal_length, bins=20, label='sepal_length', color='skyblue')
plt.hist(iris.petal_length, bins=20, label='petal_length', color='violet')
plt.legend(loc='upper right')
plt.show()

#%%

# 누적 히스토그램
plt.hist(iris.sepal_length, cumulative=True, label='cumulative', color='skyblue')
plt.hist(iris.sepal_length, cumulative=False, label='not cumulative', color='violet')
plt.legend(loc='upper left')
plt.show()


#%%

# 히스토그램 4가지 타입
fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3) 
ax4 = fig.add_subplot(2, 2, 4)

ax1.hist((iris.sepal_length, iris.petal_length), histtype='bar')
ax2.hist((iris.sepal_length, iris.petal_length), histtype='barstacked')
ax3.hist((iris.sepal_length, iris.petal_length), histtype='step')
ax4.hist((iris.sepal_length, iris.petal_length), histtype='stepfilled')

ax1.title.set_text('Type : bar')
ax2.title.set_text('Type : barstacked')
ax3.title.set_text('Type : step')
ax4.title.set_text('Type : stepfilled')

plt.tight_layout()
plt.show()

#%%

###############################################################################
# 범주형 데이터
###############################################################################

#%%

iris_petal_length_mean = iris.groupby('species').petal_length.mean()
iris_petal_length_mean

#%%
iris.species.unique()


#%%

# 막대 그래프
plt.bar(iris.species.unique(), iris.groupby('species').petal_length.mean())
plt.show()


#%%

# 막대 그래프
# 컬러 지정

plt.bar(iris.species.unique(), iris_petal_length_mean, color='mediumseagreen')
plt.show()


#%%

# 막대 그래프의 막대 마다 다른 컬러 지정
plt.bar(iris.species.unique(), iris_petal_length_mean, color=['gold', 'mediumseagreen', 'teal'], width=0.98)
plt.show()


#%%

# 수평 막대그래프
plt.barh(iris.species.unique(), iris_petal_length_mean)
plt.show()


#%%

###############################################################################
# 수치형 데이터 상관관계
###############################################################################

#%%

# 산점도
plt.scatter(x=iris.petal_length, y=iris.petal_width)
plt.show()


#%%

# 사이즈: s=area
# 컬러: c=colors
# 투명도: alpha=0.75
area = (18 * np.random.rand(len(iris.petal_length)))**2
colors = np.random.rand(len(iris.petal_length))

plt.scatter(iris.petal_length, iris.petal_width, s=area, c=colors, alpha=0.75, cmap='Set1_r')
plt.show()


#%%

# 컬러 팔레트 선택 : RdYlGn
# 컬럼맵: https://matplotlib.org/stable/tutorials/colors/colormaps.html
import matplotlib.cm as cm
colors = cm.RdYlGn(np.linspace(0, 1, iris.petal_width.shape[0]))

plt.scatter(iris.petal_length, iris.petal_width, s=10**2, c=colors)
plt.show()


#%%

setosa = iris[iris.species == 'setosa']
versicolor = iris[iris.species == 'versicolor']
virginica = iris[iris.species == 'virginica']


#%%

# 산점도: 컬러를 다르게 표시
plt.scatter(setosa.sepal_length, setosa.petal_width, color = 'hotpink')
plt.scatter(versicolor.sepal_length, versicolor.petal_width, color = '#88c999')
plt.scatter(virginica.sepal_length, virginica.petal_width, color = 'skyblue')

plt.show()


#%%

###############################################################################
# 데이터의 분포 확인과 이상값 찾기
###############################################################################

#%%

# 박스플롯
plt.boxplot([iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width])
plt.show()


#%%

# 이상값 해석
# 이상값 기준 : whis: 1.5 기본값
# notch : True, 중앙값의 95% 신뢰구간을 노치 형태로 표시
plt.boxplot([iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width], 
            notch=True, whis=2.0)
plt.show()


#%%

# 
boxplot = plt.boxplot([iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width])

whiskers = [item.get_ydata() for item in boxplot['whiskers']]
medians = [item.get_ydata() for item in boxplot['medians']]
fliers = [item.get_ydata() for item in boxplot['fliers']]

plt.show()


#%%

# 최솟값, 1사분위, 3사분위, 최댓값
print('whiskers:', whiskers)


#%%

# 중앙값
print('medians:', medians)

#%%

# 수염을 벗어나는 이상값
print('fliers:', fliers)


#%%

# 수평 박스플롯
plt.boxplot([iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width], vert=False)
plt.show()

###############################################################################
# 데이터 분포와 상세 모양
###############################################################################

#%%

# 바이올린 그래프
# 전체적인 데이터 분포 모양 파악
# positions: 각 변수의 그래프 출력 순서
plt.violinplot([iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width], positions=[2, 1, 4, 3])
plt.show()


#%%

# showmeans: True, 중앙값을 표시
plt.violinplot([iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width], showmeans=True)
plt.show()


#%%

# quantiles: 분포 분위수를 리스트 입력하여 표시
plt.violinplot([iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width], 
               quantiles=[[0.25, 0.75], [0.1, 0.9], [0.2, 0.8], [0.35, 0.65]])
plt.show()

#%%

# 컬러 변경
violinplot = plt.violinplot([iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width], showmeans=True)

violinplot['bodies'][0].set_facecolor('hotpink')
violinplot['bodies'][1].set_facecolor('purple')
violinplot['bodies'][2].set_facecolor('limegreen')
violinplot['bodies'][3].set_facecolor('darkgreen')

# 플롯의 라인 컬러 변경. 순서: 세로 라인, 최댓값, 최솟값, 중앙값 표시 라인
violinplot['cbars'].set_edgecolor('k')
violinplot['cmaxes'].set_edgecolor('darkgrey')
violinplot['cmins'].set_edgecolor('darkgrey')
violinplot['cmeans'].set_edgecolor('k')

plt.show()

