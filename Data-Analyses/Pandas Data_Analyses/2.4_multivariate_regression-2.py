# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:40:55 2024

@author: leehj
"""

# 다중회귀분석(Multivariate Regression)
# 다중회귀분석과 다항회귀분석(Polynomial Regression) 결합

### 기본 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn 라이브러리에서 선형회귀분석 모듈 가져오기
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures   #다항식 변환

#%%

'''
[Step 1 ~ 3] 데이터 준비 
'''
# CSV 파일을 데이터프레임으로 변환
df = pd.read_csv('./auto-mpg.csv', header=None)

# 열 이름 지정
df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name'] 

#%%
# horsepower 열의 자료형 변경 (문자열 ->숫자)
# df['horsepower'].replace('?', np.nan, inplace=True)      # '?'을 np.nan으로 변경
df['horsepower'] = df['horsepower'].replace({'?':np.nan})      # '?'을 np.nan으로 변경

#%%

df.dropna(subset=['horsepower'], axis=0, inplace=True)   # 누락데이터 행을 삭제
df['horsepower'] = df['horsepower'].astype('float')      # 문자열을 실수형으로 변환

# 분석에 활용할 열(속성)을 선택 (연비, 실린더, 출력, 중량)
ndf = df[['mpg', 'cylinders', 'horsepower', 'weight']]


#%%
'''
Step 4: 데이터셋 구분 - 훈련용(train data)/ 검증용(test data)
'''

# 속성(변수) 선택
X=ndf[['cylinders', 'horsepower', 'weight']]  # 훈련: 독립 변수 X1, X2, X3
y=ndf['mpg']     # 타겟: 종속 변수 Y

#%%
# train data 와 test data로 구분(7:3 비율)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10) 

print('훈련 데이터: ', X_train.shape)
print('검증 데이터: ', X_test.shape)   
print('\n') 

#%%

'''
Step 5: 다중회귀분석 모형 - sklearn 사용
'''

# 다항회귀
# 다항식 변환 
poly = PolynomialFeatures(degree=3)          # 3차항 적용
X_train_poly = poly.fit_transform(X_train)   # X_train 데이터를 2차항으로 변형

#%%

# fit과 transform 분리
# fit : 데이터를 학습
# transofrom : 학습에서 얻은 정보로 계산
# X_test_poly = poly.fit_transform(X_test) 
test_poly = PolynomialFeatures(degree=3)          # 3차항 적용
test_poly.fit(X_test) 
X_test_poly = test_poly.transform(X_test) 

#%%
# 단순회귀분석 모형 객체 생성
lr = LinearRegression()   

# train data를 가지고 모형 학습
# lr.fit(X_train, y_train)
lr.fit(X_train_poly, y_train)

# 학습을 마친 모형에 test data를 적용하여 결정계수(R-제곱) 계산
# r_square = lr.score(X_test, y_test)
r_square = lr.score(X_test_poly, y_test)
print(r_square) # 0.7744507210485588
print('\n')
#%%

# 회귀식의 기울기
print('X 변수의 계수 a: ', lr.coef_)
print('\n')

# 회귀식의 y절편
print('상수항 b', lr.intercept_)
print('\n')

#%%
# train data의 산점도와 test data로 예측한 회귀선을 그래프로 출력 
# y_hat = lr.predict(X_test)
y_hat = lr.predict(X_test_poly)

plt.figure(figsize=(10, 5))
ax1 = sns.kdeplot(y_test, label="y_test")
ax2 = sns.kdeplot(y_hat, label="y_hat", ax=ax1)
plt.legend()
plt.show()