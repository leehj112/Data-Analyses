# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:41:39 2024

@author: leehj
"""

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

url='https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt'
response=requests.get(url)
data=pd.read_csv(url,delimiter='\t')
print(data.head())

print(data.describe())
print(data.info())          #NULL값 없음

#%%



import matplotlib.pyplot as plt

plt.scatter(data['BMI'], data['BMI'], alpha=0.1)
plt.show() 

plt.figure(figsize=(9, 6))
plt.scatter(data['BMI'], data['BMI'], alpha=0.1) 
plt.show() 
# 산점도 size 크기 조정: 너비==> 9인치, 높이 => 6인치 
# data맷폴롯립 scatter()함수 시각화 ==> 산점도 측정 

fig, axs = plt.subplots(2) 

axs[0].scatter(data['BMI'],data['BMI'], alpha= 0.1)

axs[1].hist(data['BMI'], bins=100)
axs[1].set_yscale('log')

fig.show()

# subplots() 함수 
# 매개변수에 2개 use / scatter()함수, hist()함수 호출  


#%%

plt.hist(data['BMI'], bins=50)
plt.yscale('log')
plt.show() 

np.random.seed(42)
sample_means = []
for _ in range(1000):
    m = data['BMI'].sample(30).mean()
    sample_means.append(m) 
    
plt.hist(sample_means, bins=30)
plt.show() 

#%% 





"""
age (나이): 환자의 나이
sex (성별): 환자의 성별 (남성은 1, 여성은 0)
bmi (체질량 지수): 환자의 체질량 지수
bp (평균 혈압): 환자의 평균 혈압
s1 (혈청 검사 1): 총 혈청 콜레스테롤
s2 (혈청 검사 2): 저밀도 지단백
s3 (혈청 검사 3): 고밀도 지단백
s4 (혈청 검사 4): 총 콜레스테롤 / HDL 비율
s5 (혈청 검사 5): 혈청 트리글리세라이드의 로그
s6 (혈청 검사 6): 혈당 수준
y  기준 이후 1년간의 질병 진행의 양적 측정값,관심 대상인 질병 진행의 양적 지표로 사용
    당뇨병 환자의 질병 상태의 진행을 나타내며, 다양한 요인들과의 상관 관계나 예측 모델링 등의 분석에 사용.
"""

            


#%% 
#데이터 상관관계분석
correlation_matrix=data.corr()
print(correlation_matrix)

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show() 
# 그래프 

#변수분포 히스토그램
data.hist(figsize=(12,10))
plt.tight_layout()


#%% 

from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(data,random_state=42)       #75:25
print(len(train_set),len(test_set))             #331 111

x_train=train_set[['BMI']]
y_train=train_set['Y']

print(x_train.shape,y_train.shape)              #(331, 1) (331,)  ->2차원배열,1차원배열

#선형회귀모델 훈련하기
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

#훈련된 모델을 평가하기:결정계수
x_test=test_set[['BMI']]
y_test=test_set['Y']
lr.score(x_test,y_test)             

#연속적인 값 예측하기:선형회귀
print(lr.coef_,lr.intercept_)       #[10.51165308] -125.17664066850949

#카테고리 예측하기:로지스틱 회귀
y_mean=data['Y'].mean()
print(y_mean)   

y_train_c=y_train>y_mean
y_test_c=y_test>y_mean             


from sklearn.linear_model import LogisticRegression
logr=LogisticRegression()
logr.fit(x_train,y_train_c)
logr.score(x_test,y_test_c)     
    
"""
AGE       SEX       BMI  ...        S5        S6         Y
AGE  1.000000  0.173737  0.185085  ...  0.270774  0.301731  0.187889
SEX  0.173737  1.000000  0.088161  ...  0.149916  0.208133  0.043062
BMI  0.185085  0.088161  1.000000  ...  0.446157  0.388680  0.586450
BP   0.335428  0.241010  0.395411  ...  0.393480  0.390430  0.441482
S1   0.260061  0.035277  0.249777  ...  0.515503  0.325717  0.212022
S2   0.219243  0.142637  0.261170  ...  0.318357  0.290600  0.174054
S3  -0.075181 -0.379090 -0.366811  ... -0.398577 -0.273697 -0.394789
S4   0.203841  0.332115  0.413807  ...  0.617859  0.417212  0.430453
S5   0.270774  0.149916  0.446157  ...  1.000000  0.464669  0.565883
S6   0.301731  0.208133  0.388680  ...  0.464669  1.000000  0.382483
Y    0.187889  0.043062  0.586450  ...  0.565883  0.382483  1.00000
"""

"""
 331 111
(331, 1) (331,)
[10.51165308] -125.17664066850949
152.13348416289594
Out[4]: 0.7027027027027027
"""

