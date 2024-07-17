# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 23:34:33 2024

@author: dlgur
"""

# [04] 통계분석과 기본 그래프

###############################################################################
# spipy 두 집단 간 평균이 같은지 검정할 경우(독립 이표본 검정)

# 정규성 검정(사피로-윌크 검정, Shapiro-Wilk Test)
# 함수: shapiro()
# 내용: 정규분포인지 검정
#       p-value가 0.05보다 크면(귀무가설) 채택 95% 신뢰수순에서 정규분포 포함

# 독립 이표본 검정: Independent Two-Sample t-Test
# 두 집단의 평균이 통계적으로 유의미하게 다른지 판단
# 함수: ttest_ind()
# 내용: 정규분포일 경유 진행
#       p-value가 0.05보다 작으면 95% 신뢰수준에서 대립가설(두 집단 간 평균이 다름) 채택

###############################################################################


#%%

# 스토리텔링

# [분석 스토리]  
# 양계장엔 7개의 부화장이 있고, 
# 부화장 마다 최대 30개의 알을 부화시킬 수 있습니다. 
# 병아리가 부화하는데 걸리는 기간은 약 21일입니다. 
# 어제까지 딱 21일이 지났습니다. 
# 양계장에 처음으로 생명이 탄생했는데, 
# 총 몇 마리의 병아리가 부화했는지 알아보도록 하겠습니다.

# ## 어제까지 몇 마리의 병아리가 부화했을까? (기초통계량)

# ### 데이터 불러오기

# In[1]:


# pandas 패키지 불러오기 및 pd라는 약어로 지칭하기
import pandas as pd  


# In[2]:


hat = pd.read_csv('dataset/ch4-1.csv') # hat 변수에 데이터셋 입력


# ### 데이터 확인하기

# In[3]:


hat

#%%

# hatchery: 부화장
# chick: 부화된 병아리(마릿수)
"""
  hatchery  chick
0        A     30
1        B     30
2        C     29
3        D     26
4        E     24
5        F     28
6        G     27
"""


# In[4]:


hat.head() # 위에서 부터 5개 데이터 확인


# In[5]:


hat.tail(3) # 아래에서 부터 3개 데이터 확인


# ### 기초 통계량 구하기

# In[6]:


hat.chick.sum() # 합계 구하기


# In[7]:


hat['chick'].sum() # 합계 구하기


# In[8]:


hat['chick'].mean() # 평균 구하기


# In[9]:


hat['chick'].std() # 표준편차 구하기


# In[10]:


hat['chick'].median() # 중앙값 구하기


# In[11]:


hat['chick'].min() # 최소값 구하기


# In[12]:


hat['chick'].max() # 최대값 구하기


# ### 1-4. 데이터 정렬하기

# In[13]:


# 데이터 정렬하기, chick 열을 기준으로 오름차순 정렬
hat.sort_values(by=['chick'], ascending=True)


# ### 막대 그래프 그려보기

# In[14]:


# 그래프용 모듈 불러오기 및 plt라는 약어로 지칭하기
import matplotlib.pyplot as plt


# In[15]:


# 막대 그래프 그리기
plt.bar(hat['hatchery'], hat['chick'])
plt.show()


# In[16]:


# 막대 그래프 그리기, 다양한 파라미터 추가
plt.figure(figsize=(15, 10))
plt.bar(hat['hatchery'], hat['chick'], color = ('red','orange','yellow','green','blue','navy','purple'))
plt.title('부화장별 병아리 부화현황')
plt.xlabel('부화장')
plt.ylabel('부화마릿수')
plt.show()


# > pyplot 모듈에서 사용할 수 있는 함수는 아래 사이트에서 확인 가능합니다.  
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html

# ### 한글 폰트 지정 및 그래프 색상 바꿔보기

# In[17]:


# 그래프 한글 깨짐 문제 해결을 위해 맑은고딕 폰트 지정
from matplotlib import font_manager, rc
font_path = "c:/Windows/Fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# > seaborn 패키지에서 사용 가능한 팔렛트
# <div>
# <img src="https://qiita-image-store.s3.amazonaws.com/0/13955/279e0621-678a-8345-49b5-d5af7dec1544.png" width="500"/>
# </div>

# In[18]:


# 예쁜 색상 지정을 위한 seaborn 패키지 불러온 후 막대 그래프 그리기
import seaborn as sns
col7 = sns.color_palette('Pastel2', 7)
plt.figure(figsize=(10, 7))
plt.bar(hat['hatchery'], hat['chick'], color = col7, edgecolor = 'black')
plt.title('부화장별 병아리 부화현황')
plt.xlabel('부화장')
plt.ylabel('부화마릿수')
plt.show()


# ### 그래프 위에 텍스트 추가하기

# In[19]:


# 텍스트 추가 사용자 정의 함수 만들기
def addtext(x,y):
    for i in range(len(x)):
        plt.text(i,y[i]+0.5,y[i], ha = 'center')


# > ※ 기본폰트(10) 크기 일괄조정 방법  
# plt.rcParams.update({'font.size': 폰트크기})

# In[20]:


# 막대 위에 텍스트 추가하기
col7 = sns.color_palette('Pastel2', 7)
plt.figure(figsize=(10, 7))
plt.bar(hat['hatchery'], hat['chick'], color = col7, edgecolor = 'black')
addtext(hat['hatchery'], hat['chick']) # 텍스트 표시 사용자 정의 함수 추가
plt.title('부화장별 병아리 부화현황', fontsize =17)
plt.xlabel('부화장')
plt.ylabel('부화마릿수')
plt.show()


#%%

# ### 그래프 위에 선 추가하기

# In[21]:


# 빨간색 수평선 추가하기
col7 = sns.color_palette('Pastel2', 7)
plt.figure(figsize=(10, 7))
plt.bar(hat['hatchery'], hat['chick'], color = col7, edgecolor = 'black')
addtext(hat['hatchery'], hat['chick']) # 텍스트 표시 사용자 정의 함수 추가
plt.hlines(30, -1, 7, colors = 'red', linestyles = 'dashed')
plt.title('부화장별 병아리 부화현황', fontsize =17)
plt.xlabel('부화장')
plt.ylabel('부화마릿수')
plt.show()


# ### 파이 차트 그려보기

# In[22]:


# 파이차트를 그리기 위해 비율 계산
pct = hat['chick']/hat['chick'].sum()
pct


# In[23]:


# 파이차트 그리기
col7 = sns.color_palette('Pastel2', 7)
plt.figure(figsize=(10, 10))
plt.pie(pct, labels = hat['hatchery'], autopct='%.1f%%', colors = col7, startangle = 90, counterclock = False)
plt.show()


#%%

# [분석 스토리]  
# 기초 통계량 및 기본 그래프를 통해 어제까지 194마리의 병아리가 부화했고, 
# 부화장별 평균 약 28마리가 태어난 것을 확인했으며 
# 부화장 A, B에서는 모든 알이 병아리로 부화한 것을 알 수 있게 되었습니다. 
# 아마도 2~3일 정도 더 기다린다면 나머지 부화장의 알도 대부분 부화할 것 입니다.
# 

# -----------------------------------------------------------------------------

# ## 부화된 병아리들의 체중은 얼마일까? (정규분포와 중심극한정리)

# 정규분포:
# 평균과 표준편차에 의해 모양이 결정되는 연속확률분포
    
# 중심극한정리:
# 데이터가 적당히 많을 경우(일반적으로 30건 이상) 정규분포에 가까워진다.
# 중심극한정리를 이용하면 평균과 편차만 알고 있어도 대략적인 데이터의 분포를 알아 낼 수 있다.

# [분석 스토리]  
# 체계적인 사육을 위해 부화된 병아리 모두에 개별 데이터를 수집
# 병아리들의 몸무게를 측정

# ### 데이터 불러와서 구조와 유형 확인하기

# In[24]:


# pandas 패키지 불러오기 및 pd라는 약어로 지칭하기
import pandas as pd  


# In[25]:


b = pd.read_csv('dataset/ch4-2.csv') # b 변수에 데이터셋 입력


# In[26]:


b.head() # 데이터 상위 5개만 확인


# In[27]:


b.info() # b 데이터셋 정보 확인


# ### 통계량으로 분포 확인하기

# In[28]:


b.describe() # b 데이터셋 기초통계량 확인

#%%

"""
          weight
count  30.000000
mean   38.400000
std     3.286335
min    31.000000
25%    36.250000
50%    39.000000
75%    40.750000
max    45.000000
"""

#%%

# 평균과 표준편차를 이용한 병아리 몸무게 분포 추정
# https://en.wikipedia.org/wiki/Box_plot
# 몸무게 평균: 38.4
# 표준편차: 3.286335
# 평균(u)
# 표준편차(s)
"""
구분            최솟값        1사분위       3사분위     최대값
-------------------------------------------------------------------------------
실제값           31.00          36.25         40.75      45.00
-------------------------------------------------------------------------------
추정값           28.54          36.18         40.65      48.26
-------------------------------------------------------------------------------
추정값 계산식    u-3s       u-0.6745s     u+0.6745s       u+3q
-------------------------------------------------------------------------------


"""

#%%



# ※ pandas 패키지에서 std() 메서드는 표본 표준편차를 기본값으로 계산하며, 
# numpy 모듈에서 std() 함수는 모 표준편차를 기본값으로 계산합니다. 

# 이 두 함수 모두 ddof 파라미터를 적용할 수 있는데 
# 이 값을 0으로 지정하면 모 표준편차, 1로 지정하면 표본 표준편차가 됩니다. 

# [모 표준편차와 표본 표준편차]
# 모 표준편차 (σ): 전체 모집단의 표준편차
# 표본 표준편차 (s): 모집단의 일부인 표본의 표준편차
# 목적:
#   - 모 표준편차는 전체 모집단의 변동성을 정확하게 측정하는 데 사용
#   - 표본 표준편차는 주어진 표본을 통해 모집단의 표준편차를 추정하는 데 사용
#   - 표본 표준편차는 모집단의 표준편차보다 약간 더 크게 계산되는 경향이 있으며,
#   - 이는 작은 표본 크기로 인해 발생할 수 있는 추정 오차를 보정하기 위해서입니다.

#%%

# 넘파이 예제
# ddof(Delta Degrees of Freedom)는 자유도를 의미
# 모 표준편차 : ddof=0
# 표본 표준편차: ddof=1

import numpy as np

# 예제 데이터 (모집단)
data = [10, 12, 23, 23, 16, 23, 21, 16]

# 모 표준편차 계산 (ddof=0)
population_std = np.std(data, ddof=0)
print(f"모 표준편차: {population_std}") # 4.898979485566356

# 표본 표준편차 계산 (ddof=1)
sample_std = np.std(data, ddof=1)
print(f"표본 표준편차: {sample_std}") # 5.237229365663817

#%%

# 판다스 예제
import pandas as pd

# 예제 데이터 (Series로 생성)
data = pd.Series([10, 12, 23, 23, 16, 23, 21, 16])

# 모 표준편차 계산 (ddof=0)
population_std = data.std(ddof=0)
print(f"모 표준편차: {population_std}") # 4.898979485566356

# 표본 표준편차 계산 (ddof=1)
sample_std = data.std(ddof=1)
print(f"표본 표준편차: {sample_std}") # 5.237229365663817

#%%
# ### 히스토그램으로 분포 확인하기

# In[29]:


# 그래프용 모듈 불러오기 및 plt라는 약어로 지칭하기
import matplotlib.pyplot as plt


# In[30]:


# 히스토그램 그리기
plt.figure(figsize=(10, 7))
plt.hist(b.weight, bins = 7)
plt.title('B 부화장 병아리 무게 분포 현황', fontsize =17)
plt.xlabel('병아리 무게(g)')
plt.ylabel('마릿수')
plt.show()


# ### 상자그림으로 분포 확인하기

# In[31]:


# 상자그림 그리기
plt.figure(figsize=(8, 10))
plt.boxplot(b.weight)
plt.title('B 부화장 병아리 무게 상자그림', fontsize =17)
plt.ylabel('병아리 무게(g)')
plt.show()


# ### 다중 그래프로 분포 확인하기

# In[32]:


# 히스토그램과 상자그림 한 번에 표시
plt.figure(figsize=(10, 12))
plt.subplot(2, 1, 1)
plt.hist(b.weight, bins = 7)
plt.title('B 부화장 병아리 무게 분포 현황', fontsize = 17)
plt.subplot(2, 1, 2)
plt.boxplot(b.weight, vert = False)
plt.show()


#%%

# [분석 스토리]  
# 히스토그램과 상자그림을 통해 병아리 몸무게가 어느 정도인지 확인한 결과 30마리의 체중이 30 ~ 45g 사이에 분포하며 
# 그 중 절반은 36.25(1사분위수) ~ 40.75(3사분위수)g 사이에 분포하고 있음을 알 수 있었습니다. 
# 게다가 중심극한정리를 이용해 평균과 표준편차만으로도 대략적인 몸무게의 분포를 추정할 수 있음을 알게 되었습니다.

# ## 사료 제조사별 성능차이가 있을까? (가설검정)

# [분석 스토리]  
# 병아리가 부화한 지 5일이 지났습니다. 
# 그런데 이상한 점을 발견했습니다. 
# 부화장 A에서 태어난 병아리 대비 부화장 B에서 태어난 병아리의 덩치가 더 작아 보입니다. 
# 서로 다른 사료를 먹이고 있긴 한데 기분 탓인지, 아니면 정말 작은지 한 번 검정해 보겠습니다.

# ### 데이터 불러와서 확인하기

# In[33]:


# pandas 패키지 불러오기 및 pd라는 약어로 지칭하기
import pandas as pd


# In[34]:


test = pd.read_csv('dataset/ch4-3.csv') # test 변수에 데이터셋 입력
test


# ### 상자그림으로 분포 비교하기

# In[35]:


# seaborn 패키지 이용 그룹별 상자그림 그리기
import seaborn as sns
plt.figure(figsize=(10, 7))
sns.boxplot(x = 'weight', y = 'hatchery', data= test)
plt.title('부화장 A vs. B 몸무게 분포 비교', fontsize = 17)
plt.show()

#%%

###############################################################################
# ### 정규 분포인지 검정하기
###############################################################################
# 두 집단 간의 몸무게 평균이 같은지 다른지 가설검정의 방법론인 t-test를 통해 진행
# t-test는 데이터가 정규분포를 한다는 가정하게 평균이 데이터의 대표값 역할을 한다고 전제
# 따라서 t-test를 수행하기 전에 데이터가 정규분포를 따른는지 사피로-윌크 검정(Shapiro-Wilk Test)를 통해 판정
#
# [결과해석]
# 사피로-윌크 검정에서 귀무가설은 '정규분포한다'고, 
# 대립가설은 '정규분포하지 않는다'이다.
# p-value: test_a는 0.555, test_b는 0.5427

# In[36]:


# 가설검정을 위한 scipy 패키지 불러오기 및 sp로 지칭하기
import scipy as sp


# In[37]:


# 부화장 A만 별도로 데이터셋 구성
test_a = test.loc[test.hatchery == 'A', 'weight']


# In[38]:


# 부화장 B만 별도로 데이터셋 구성
test_b = test.loc[test.hatchery == 'B', 'weight']


# In[39]:


# 부화장 A: 샤피로 윌크 검정 실시
sp.stats.shapiro(test_a)

# ShapiroResult(statistic=0.9400016973451231, pvalue=0.5530322552073004)
# → p값이 0.553으로 유의수준 0.05보다 크기 때문에 귀무가설 채택(정규분포함)

# In[40]:


# 부화장 B: 샤피로 윌크 검정 실시
sp.stats.shapiro(test_b)

# ShapiroResult(statistic=0.9390683777600799, pvalue=0.5426943326835627)
# → p값이 0.543으로 유의수준 0.05보다 크기 때문에 귀무가설 채택(정규분포함)

# > ※ 신뢰수준, 유의수준, p-value  
# >* 신뢰수준(1-α) : 통계에서 어떠한 값이 알맞은 추정값이라고 믿을 수 있는 정도를 뜻하며 주로 95%를 사용합니다. 
#    신뢰도라고도 부릅니다.
# >* 유의수준(α) : 통계적인 가설검정에서 사용되는 기준값을 말합니다.
# >* p-value(유의확률) : 귀무가설이 맞다고 가정할 때 얻은 결과보다 극단적인 결과가 실제로 관측될 확률을 말합니다. 
#    p값이 유의수준 보다 작으면 귀무가설을 기각하고, 대립가설을 채택하며, 반대일 경우에는 귀무가설을 채택합니다. 
# 

#%%

from scipy import stats

# 예제 데이터
g1 = [20, 21, 19, 18, 22, 20, 21]
g2 = [30, 29, 31, 32, 33, 30, 29]

# 독립 이표본 t-검정 수행
t_stat, p_value = stats.ttest_ind(g1, g2)

print(f"t-통계량: {t_stat}") # -13.63434445938672
print(f"p-값: {p_value}")    # 1.151824511148725e-08

# 결과 해석
alpha = 0.05
if p_value < alpha:
    print("귀무가설 기각: 두 집단의 평균은 유의미하게 다릅니다.")
else:
    print("귀무가설 채택: 두 집단의 평균은 유의미하게 다르지 않습니다.")
    

#%%
# ### t-test로 두 집단 간 평균 검정하기


# In[41]:

# 부화장 A, B 두 집단 간 평균 검정
sp.stats.ttest_ind(test_a, test_b)

# TtestResult(statistic=2.842528280230058, pvalue=0.010803990633924204, df=18.0)

# → p값이 0.01로 0.05보다 작기 때문에 95% 신뢰수준에서 대립가설 채택, 즉 두 집단 간 평균은 서로 다르다고 판단됨

#%%

# [분석 스토리]  
# 부화장 B의 병아리들이 부화장 A의 병아리들보다 덩치가 작았던 것은 기분 탓이 아니었다.
# 납품기한 문제로 인해 불가피하게 수급한 B사의 사료 품질이 A사 대비 떨어졌기 때문이다.
# B사와 거래를 끊고, 며칠 간 발품을 팔아 새로운 사료 제조사인 C사와 거래
# A사와 동일한 품질의 사료임을 t-test를 통해 판정할 수 있었음.
# 
# 그 후 사료 수급처를 다변화하면서 안정적인 사료 공급망을 구축해 위기를 모면
# 기본적인 병아리 생산 현황 파악 및 
# 사료 문제를 해결한 후 본격적으로 병아리가 건강하게 성장할 수 있는 방법을 모색하기 시작
# 

#%%

# THE END