# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 23:35:44 2024

@author: dlgur
"""

# # [05] 상관분석과 회귀분석

#%%
# 상관분석(Correlation Analysis)
# 상관분석과 회귀분석은 데이터 분석 모델을 만들기 위한 가장 기초적인 관문이다.
# 상관분석은 다양한 변수가 서로 비례 관계인지 반비례 관계인지를 부호(±)와 숫자로 표현

# 연속형인 두 변수 간에 어떤 선형적인(linear) 또는 비선형적인(non-linear) 관계를 갖고 있는지 분석하는 방법이다.
# 상관분석을 하면 두 변수 간의 관계를 상관계수(Correlation Cofficient)로 나타낸다.
# 상관계수는 -1과 1사이 값을 갖는다.
# 부호(-) : 반비례 관계인 음의 상관관계를 갖는다.
# 부호(+) : 비례 관계인 양의 상관관계를 갖는다.
# -1.0 ~ -0.7 : 강한 음의 관계
# -0.7 ~ -0.3 : 뚜렷한 음의 상관관계
# -0.3 ~ -0.1 : 약한 음의 관계
# -0.1 ~ +0.1 : 거의 무시됨
# +0.1 ~ +0.3 : 약한 양의 상관관계
# +0.3 ~ +0.7 : 뚜렷한 양의 상관관계
# +0.7 ~ +1.0 : 강한 양의 상관관계
# ※ 상관계수는 두 변수 간에 연관된 정도만을 나타내며, 인관관계를 설명하는 것은 아니다.

#%%

# 회귀분석(Regression analysis)
# 회귀분석은 서로 상관관계가 있는 연속형 변수들의 관계를 수식으로 나타냄
# 독립변수(x)와 종속변수(y)가 존재할 때 두 변수 간의 관계를 y = ax + b 형태의 수식으로 나타낼 수 있는 방법

# 회귀분석의 5가지 가정
# 회귀분석은 5가지 가정을 전제로 한다.
#   1. 선형성: 독립변수(x)와 종속변수(y)의 관계가선형 관계가 있음
#   2. 독립성: 잔차(residual)와 독립변수의 값이 관련이 없어야 함
#   3. 등분산성: 독립변수의 모든 값에 대한 오차들의 분산이 일정해야 함
#   4. 비상관성: 관측치들의 잔차들끼리 상관이 없어야 함
#   5. 정상성: 잔차항이 정규분포를 이뤄야 함

# 단순선형 회귀분석
#   - 종속변수(y)와 독립변수(x)가 각각 하나씩 존재하며 서로 선형적인 관계를 가짐
#   - 회귀모델(모형 도는 식)은 y = ax + b 형태의 수식으로 나타냄

#%%

# [분석 스토리]  
# 이제 막 병아리를 키우고 있는 병아리의 성장 속도에 영향을 미친 인자들은 무엇인가?
# 병아리의 성장 속도가 빠르면 보다 많은 매출을 올릴 수 있기 때문이다.
# 아마도 유전적인 요소도 중요할 것이고, 사료를 얼마만큼 먹는지도 중요할 것이다.
# 

# ## 병아리의 성장에 영향을 미치는 인자는 무엇일까? (상관분석)

# ### 데이터 불러와서 확인하기

# In[1]:


# pandas 패키지 불러오기 및 pd라는 약어로 지칭하기
import pandas as pd


# In[2]:


w = pd.read_csv('dataset/ch5-1.csv') # w 변수에 데이터셋 입력


# In[3]:

# 부화된지 1주일된 병아리
# weight: 몸무게
# egg_weight: 종란 무게
# movement: 하루 평균 이동거리
# food: 하루 평균 사료 섭취량


w.head() # 위에서 부터 5개 데이터 확인


# In[4]:


w.info() # 데이터 구조 및 자료형 확인

#%%

"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 30 entries, 0 to 29
Data columns (total 5 columns):
 #   Column      Non-Null Count  Dtype 
---  ------      --------------  ----- 
 0   chick_nm    30 non-null     object
 1   weight      30 non-null     int64 
 2   egg_weight  30 non-null     int64 
 3   movement    30 non-null     int64 
 4   food        30 non-null     int64 
dtypes: int64(4), object(1)
memory usage: 1.3+ KB
"""


#%%
# ### 상관분석을 위한 별도 데이터 셋 만들기

# In[5]:


# w 데이터셋에서 1~4열 데이터만 가져오기
w_n = w.iloc[:,1:5]
w_n.head()


# ### 상관분석 실시

# In[6]:


# 상관분석 실시
# 피어슨(Pearson) 상관계수는 선형적인 관계의 크기 측정
# ※ 스피어만(Spearman) 상관계수는 두 변수가 순서 또는 서열 척도인 경우와 비선형적인 관계의 크기 측정
w_cor = w_n.corr(method = 'pearson')
w_cor

#%%

# 상관행렬(corrlation Matrix) 형태
"""
              weight  egg_weight  movement      food
weight      1.000000    0.957169  0.380719  0.877574
egg_weight  0.957169    1.000000  0.428246  0.808147
movement    0.380719    0.428246  1.000000  0.319011
food        0.877574    0.808147  0.319011  1.000000

"""
# 결과
# 병아리 몸무게(weight)와 가장 큰 상관계수를 갖는 변수?
#   - 종란무게(egg_weight) : 0.957169 
#   - 하루 평균 사료 섭취량(food) : 0.877574



#%%

# ### 상관분석 결과 표현하기

# In[7]:


# 상관관계 시각화를 위한 패키지 불러오기
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:

# 산점도 그리기

# 첫 번째 행의 병아리 몸무게(weight)를 기준으로 확인
# 병아리 몸부게가 y축일 때 나머지 변수들이 x축인 경우를 나타내는 산점도로 
# 종란 무게와 하루 평균 사료 섭취량이 병아리 몸무게에 강한 양의 선형관계를 갖는다.
# 하루 평균 이동거리는 데이터의 분포가 매우 흩어져 있다.

sns.pairplot(w_n)


# In[9]:


# 상관행렬도 그리기
plt.figure(figsize = (10,7))
sns.heatmap(w_cor, annot = True, cmap = 'Blues')
plt.show()


#%%

###############################################################################
# 회귀분석
###############################################################################

# ## 병아리의 무게를 예측할 수 있을까?

# [분석 스토리]  
# 상관분석을 통해 병아리 무게에 영향을 미치는 인자들을 찾을 수 있었고, 
# 그 중에서도 병아리가 태어난 달걀인 종란의 무게가 가장 큰 양의 상관관계를 가지고 있음을 확인할 수 있다.
# 그렇다면 종란 무게로 병아리의 무게를 예측하는 게 가능한가?
# 


#%%
# ### 단순 선형 회귀분석

# 파이썬 종류
# sklearn.linear_model 모듈의 LinearRegression() 함수
# statsmodels.formula.api 모듈의 ols() 함수

# In[10]:


# 회귀분석 수행을 위한 모듈 불러오기 및 smf로 지칭하기
import statsmodels.formula.api as smf

# ols 함수
# ols(formula='y~x1 + x2 + x3 + ...', data=dataset)
# 종란무게 - 병아리 몸무게 단순선형회귀모델 구축
model_lm = smf.ols(formula = 'weight ~ egg_weight', data = w_n)


# In[11]:


# 모델 학습
result_lm = model_lm.fit()


# In[12]:


# 모델 결과 확인
result_lm.summary()


# In[13]:


# 보고서 형태로 모델 결과 출력
print(result_lm.summary())

#%%

"""
  OLS Regression Results                            
==============================================================================
Dep. Variable:                 weight   R-squared:                       0.916
Model:                            OLS   Adj. R-squared:                  0.913
Method:                 Least Squares   F-statistic:                     306.0
Date:                Tue, 25 Jun 2024   Prob (F-statistic):           1.32e-16
Time:                        11:48:03   Log-Likelihood:                -63.148
No. Observations:                  30   AIC:                             130.3
Df Residuals:                      28   BIC:                             133.1
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept    -14.5475      8.705     -1.671      0.106     -32.380       3.285
egg_weight     2.3371      0.134     17.493      0.000       2.063       2.611
==============================================================================
Omnibus:                       15.078   Durbin-Watson:                   1.998
Prob(Omnibus):                  0.001   Jarque-Bera (JB):                2.750
Skew:                           0.032   Prob(JB):                        0.253
Kurtosis:                       1.518   Cond. No.                     1.51e+03
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.51e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
"""

# Prob (F-statistic): 1.32e-16
# egg_weight, P>|t| : 0.000
# R-squared: 0.916
# coef : Intercept(-14.5475), egg_weight(2.3371)

# Prob (F-statistic): 회귀모델일 통계적으로 유의한지 확인
#   - F 통계량의 p-value가 0.05보다 작으면 유의수준 5%(신뢰수준 95%)하에서 추정된 회귀모델이 통계적이로 유의한 것으로 판단
#   - 값(1.32e-16)은 0.05보다 0에 가까운 매우 작은 값으로 회귀모델이 통계적으로 유의하다고 판단

# egg_weight, P>|t| : 개별 독립변수가 통계적으로 유의한지 확인
#   - 개별 독립변수의 값이 0.05보다 작으면 유의수준 5%하에서 통계적으로 유의한 것으로 판단
#   - 종란의 무게(egg_weight)가 0.000으로 0.05보다 작은 0에 가까운 값으로 종란 무게는 통계적으로 유의하다고 판단
#   - 상수(Intercept)의 p 값은 의미가 없다.

# R-squared: 결정계수가 높은지 확인
#   - R²는 1에 가까울수록 회귀모델의 성능(설명력)이 뛰어나나고 판단
#   - R²는 일반적으로 0.7보다 크면 우수한 회귀모델이라고 판단할 수 있다.
#   - 물론 독립변수와 종속변수의 절대적인 수치 크기로 인해 R²이 0.7보다 작더라도 효용 있는 회귀모델일 수 있다.
#   - R-squared(0.916)은 1에 가까운 매우 높은 값으로 회귀모델의 성능이 뛰어나다고 판단

# coef : coefficient 값으로 구할 수 있다.
#   - Intercept(-14.5475)는 y절편(상수)를 의미
#   - 각 독립변수에 해당되는 coef 값은 해당 독립변수의 계수(기울기)를 의미
#   - weight = 2.3371 * egg_weight - 14.5475


# In[14]:


# 산점도
# 종란무게에 따른 병아리 몸무게 산점도
# 가로축: 종란무게(egg_weight)
# 세로축: 몸무게(weight)
# 라인선: 회귀직선
plt.figure(figsize = (10,7))
plt.scatter(w.egg_weight, w.weight, alpha = .5)
plt.plot(w.egg_weight, w.egg_weight*2.3371 - 14.5475, color = 'red')
plt.text(66, 132, 'weight = 2.3371 * egg_weight - 14.5475', fontsize = 12)
plt.title('Scatter Plot')
plt.xlabel('egg_weight')
plt.ylabel('weight')
plt.show()


# In[15]:


# 잔차(residual) 5개만 확인
result_lm.resid.head()


# In[16]:


# 전차 히스토그램 그리기
plt.figure(figsize = (10,7))
plt.hist(result_lm.resid, bins = 7)
plt.show()

# 결과분석:
#   - 잔차가 0 근처에 주로 분포해 세로가 긴 종 모양의 히스트그램이 아님
#   - 잔차가 다양하게 분포

# 과제:
#   - 독립변수를 더 늘려서 회귀분석을 하라.

#%%

###############################################################################
# ### 다중 회귀분석
###############################################################################

# 다중 회귀분석(Multiple Regression Analysis)
#   - 독립변수가 2개 이상일 경우
#   - 수식: y = ax1 + bx2 + c

# In[17]:


# 병아리 몸무게 예측을 위한 다중회귀분석 실시
model_mlm = smf.ols(formula = 'weight ~ egg_weight + food + movement', data = w_n)


# In[18]:


result_mlm = model_mlm.fit()


# In[19]:


result_mlm.summary()

#%%
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                 weight   R-squared:                       0.948
Model:                            OLS   Adj. R-squared:                  0.942
Method:                 Least Squares   F-statistic:                     157.7
Date:                Tue, 25 Jun 2024   Prob (F-statistic):           8.46e-17
Time:                        12:36:21   Log-Likelihood:                -56.008
No. Observations:                  30   AIC:                             120.0
Df Residuals:                      26   BIC:                             125.6
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      2.9748      8.587      0.346      0.732     -14.676      20.626
egg_weight     1.7763      0.195      9.117      0.000       1.376       2.177
food           1.5847      0.405      3.915      0.001       0.753       2.417
movement      -0.0087      0.017     -0.522      0.606      -0.043       0.026
==============================================================================
Omnibus:                        1.993   Durbin-Watson:                   2.030
Prob(Omnibus):                  0.369   Jarque-Bera (JB):                1.746
Skew:                          -0.480   Prob(JB):                        0.418
Kurtosis:                       2.311   Cond. No.                     4.31e+03
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 4.31e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
"""

# 결과 해석은 단순 선형 회귀분석과 동일
# 다중 회귀분석에서는 개별 독립변수의 p값을 더 유심히 봐야 한다.
# Adj. R-squared로 모델이 계산을 통해 얼마나 종속변수를 잘 설명하는 확인
# 이동거리(movement)의 p-value : 0.606 > 0.05, 95% 신뢰수준에서 통계적으로 유의하지 않음, 이 독립변수는 회귀분석에서 제외

# 종란무게만으로 실시한 단순 선형 회귀분석에서 R-equared는 0.916
# 변수를 2개 추가한 다중 회귀분석에서 Adj. R-squared(0.942)로 더 높아짐

# 결론적으로 하루 평균 이동거리는 제외하고 다중 회귀분석을 실시해서 다시 평가하는 하는 것이 좋음


# In[20]:


# 병아리 몸무게 예측을 위한 다중회귀분석 실시2
model_mlm2 = smf.ols(formula = 'weight ~ egg_weight + food', data = w_n)


# In[21]:


result_mlm2 = model_mlm2.fit()


# In[22]:


result_mlm2.summary()

#%%

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                 weight   R-squared:                       0.947
Model:                            OLS   Adj. R-squared:                  0.943
Method:                 Least Squares   F-statistic:                     243.0
Date:                Tue, 25 Jun 2024   Prob (F-statistic):           5.44e-18
Time:                        12:48:45   Log-Likelihood:                -56.164
No. Observations:                  30   AIC:                             118.3
Df Residuals:                      27   BIC:                             122.5
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      3.6638      8.370      0.438      0.665     -13.510      20.837
egg_weight     1.7453      0.183      9.536      0.000       1.370       2.121
food           1.5955      0.399      4.001      0.000       0.777       2.414
==============================================================================
Omnibus:                        2.302   Durbin-Watson:                   2.103
Prob(Omnibus):                  0.316   Jarque-Bera (JB):                1.940
Skew:                          -0.502   Prob(JB):                        0.379
Kurtosis:                       2.263   Cond. No.                     1.84e+03
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.84e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
"""

# 결과: 독립변수가 하나 줄어 회귀모델은 더 간단해짐
# Adj. R-squared(0.943)가 하루 평균 이동거리(movement) 변수가 포함되었을 때 
# Adj. R-squared(0.942) 보다 올랐다.

# 다중 회귀분석에서 변수를 선택하는 방법
#   - 전진선택법(Forward) Selection): y 절편만 있는 상수모형부터 시작해 독립변수를 추가해 나감
#   - 후진소거법(Backward Elimination): 독립변수를 모두 포함한 상태에서 가정 적은 영향을 주는 변수를 하나씩 제거해 나감
#   - 단계적방법(Setpwise): y 절편만 있는 상수모형부터 시작해 독립변수를 추가해 나가지만 
#     추가한 독립변수가 중요하지 않으면(p-value가 높으면) 제거하고, 다른 독립변수를 추가해 나감

# weight = 1.7453 * egg_weight + 1.5955 * food + Intercept(3.6638)

#%%

###############################################################################
# ### 다중공선성
###############################################################################

# 다중공선성 문제(Multi-Collinearity)
# 다중 회귀분석의 경우 단순 선형 회귀분석과 달리 독립변수가 많기 때문에
# 예상치 못한 독립변수들 간의 강한 상관관계로 인해 제대로 된 회귀분석이 안 되는 경우

# 다중공선성 문제는 분산팽창요인(VIF, Varinace Inflation Factor)을 계산해 구하는데 
# 일반적으로 10 이상이면 다중공선성 문제가 있다고 판단하며
# 30 이상이면 심각한 다중공선성 문제가 있다고 판단


# In[23]:


# 다중공선성 확인을 위한 함수 불러오기
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[24]:


# 회귀모델 외생변수이름 속성
model_mlm2.exog_names


# In[25]:


# 첫 번째 변수(egg_weight) vif 계산
vif1 = variance_inflation_factor(model_mlm2.exog, 1)


# In[26]:


# 두 번째 변수(food) vif 계산
vif2 = variance_inflation_factor(model_mlm2.exog, 2)


# In[27]:


print(vif1, vif2) # 2.8826845113075725 2.8826845113075756


#%%

# 결과:
#   - 종란 무게(egg_weight) : 2.8826845113075725
#   - 하루 평균 사료 섭취량(food) : 2.8826845113075756
#   - 위 두 변수 모두 2.88 수준으로 10보다 작기 때문에 다중공선성 문제는 업는 것으로 판단
 
#%%

# In[28]:


# 잔차 히스토그램 그리기
plt.figure(figsize = (10,7))
plt.hist(result_mlm2.resid, bins = 7)
plt.show()


#%%

# [분석 스토리]  
# 다중 회귀분석을 이용해 종란 무게와 하루 평균 사료 섭취량 데이터로 병아리의 몸무게를 
# 매우 높은 정확도로 예측할 수 있는 회귀모델을 개발할 수 있었다. 
# 하지만 이 수식은 단지 부화한 지 1주일된 병아리의 몸무게를 예측하는 데 외에는 사용할 수 없었다.
# 단지 1주일된 병아리의 몸무게가 아닌 병아리가 닭이 될 때까지 성장기간에 따른 몸무게 변화가 궁금하다.
# 그래서 병아리 한 마리를 지정해 부화한 첫날부터 70일까지의 몸무게를 기록했다. 
# 성장기간에 따른 병아리의 몸무게는 과연 어떻게 변화했을까?
# 

#%%

###############################################################################
# ### 비선형 회귀분석
###############################################################################

# 비선형 회귀분석(Non-linear Regression Analysis)은 독립변수(x)와 종속변수(y)가 
# 선형 관계가 아닌 비선형 관계일 때 사용하는 분석방법이다.
# 독립변수와 종속변수가 직선이 아닌 곡선 형태의 관계를 가질 수도 있기 때문에
# 이럴 때는 독립변수에 로그(log)나 거듭제공 등을 취해 보면서 적합한 비선형 모델을 찾아내야 한다.

# In[29]:

# 병아리의 몸무게의 변화
# 변수: 성장기간(day), 병아리 무게(weight)
w2 = pd.read_csv('dataset/ch5-2.csv') # w2 변수에 데이터셋 입력


# In[30]:


w2.head()


# In[31]:


w2.info()


# In[32]:


# 성장기간에 따른 몸무게 변화
plt.figure(figsize = (10,7))
plt.scatter(w2.day, w2.weight, alpha = .5)
plt.title('Scatter Plot')
plt.xlabel('day')
plt.ylabel('weight')
plt.show()


# In[33]:


# 성장기간에 따른 병아리의 몸무게 변환 선형 회귀분석 실시
model_lm2 = smf.ols(formula = 'weight ~ day', data = w2)


# In[34]:


result_lm2 = model_lm2.fit()


# In[35]:


result_lm2.summary()

#%%

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                 weight   R-squared:                       0.979
Model:                            OLS   Adj. R-squared:                  0.979
Method:                 Least Squares   F-statistic:                     3189.
Date:                Tue, 25 Jun 2024   Prob (F-statistic):           7.22e-59
Time:                        13:53:54   Log-Likelihood:                -457.86
No. Observations:                  70   AIC:                             919.7
Df Residuals:                      68   BIC:                             924.2
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept   -295.8671     41.102     -7.198      0.000    -377.885    -213.850
day           56.8216      1.006     56.470      0.000      54.814      58.830
==============================================================================
Omnibus:                        3.866   Durbin-Watson:                   0.025
Prob(Omnibus):                  0.145   Jarque-Bera (JB):                2.079
Skew:                          -0.133   Prob(JB):                        0.354
Kurtosis:                       2.199   Cond. No.                         82.6
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# 결과:
# R-squared: 0.979    
# day: p-value(0.000) < 0.05


#%%


# In[36]:

# 산점도와 회귀직선
# 성장기간에 따른 몸무게 변화
plt.figure(figsize = (10,7))
plt.scatter(w2.day, w2.weight, alpha = .5)
plt.plot(w2.day, w2.day*56.8216 - 295.8671, color = 'red')
plt.text(40, 500, 'weight = 56.8216day - 295.8671', fontsize = 12)
plt.title('Scatter Plot')
plt.xlabel('day')
plt.ylabel('weight')
plt.show()

# 결과:
# 산점도는 3차원 함수 그래프와 유사한 형태
# 회귀직선과 편차가 존재하는 구간이 존재

# In[37]:

# 독립변수인 성장기간(day)을 3제곱하여 종속변수인 몸무게(weight) 잘 표현하는지 확인
# 개별 독립변수의 값에 제공을 취하기 위해서 I() 함수 사용하고 제곱(**)을 하여 비 선형회귀분석 실시

# 성장기간에 따른 병아리의 몸무게 변환 비선형 회귀분석 실시
model_nlm = smf.ols(formula = 'weight ~ I(day**3) + I(day**2) + day', data = w2)


# In[38]:


result_nlm = model_nlm.fit()


# In[39]:


result_nlm.summary()


#%%

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                 weight   R-squared:                       1.000
Model:                            OLS   Adj. R-squared:                  0.999
Method:                 Least Squares   F-statistic:                 4.407e+04
Date:                Tue, 25 Jun 2024   Prob (F-statistic):          7.13e-109
Time:                        14:01:02   Log-Likelihood:                -327.17
No. Observations:                  70   AIC:                             662.3
Df Residuals:                      66   BIC:                             671.3
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
===============================================================================
                  coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------
Intercept     117.0141     13.476      8.683      0.000      90.108     143.920
I(day ** 3)    -0.0253      0.000    -51.312      0.000      -0.026      -0.024
I(day ** 2)     2.6241      0.053     49.314      0.000       2.518       2.730
day           -15.2978      1.632     -9.373      0.000     -18.557     -12.039
==============================================================================
Omnibus:                        6.702   Durbin-Watson:                   0.082
Prob(Omnibus):                  0.035   Jarque-Bera (JB):                2.680
Skew:                           0.103   Prob(JB):                        0.262
Kurtosis:                       2.064   Cond. No.                     5.65e+05
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 5.65e+05. This might indicate that there are
strong multicollinearity or other numerical problems.
"""

# 결과:
# R-squared: 1.000    
# 

# In[40]:

# 산점도와 회귀곡선
# 성장기간에 따른 몸무게 변화
plt.figure(figsize = (10,7))
plt.scatter(w2.day, w2.weight, alpha = .5)
plt.plot(w2.day, (w2.day**3)*(-0.0253) + (w2.day**2)*2.6241 + w2.day*(-15.2978) + 117.0141, color = 'red')
plt.text(0, 3200, 'weight = -0.0253(day^3) + 2.6241(day^2) - 15.2978day + 117.0141', fontsize = 12)
plt.title('Scatter Plot')
plt.xlabel('day')
plt.ylabel('weight')
plt.show()

# 결과 : 거의 일치
# 회귀모델 식: weight = -0.0253 * day³ + 2.6241 * day² - 15.2978 * day + 117.0141 

#%%

# 결론: 
#   회귀분석은 데이터의 형태를 보고 그에 적합한 회귀모델을 만들어 나가는
#   작업의 반복을 통해 높은 성능을 가진 모델을 만들 수 있다.
     
    
# THE END