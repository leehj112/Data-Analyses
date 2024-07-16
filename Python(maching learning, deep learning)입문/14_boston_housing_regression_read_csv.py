# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 09:13:13 2024

@author: leehj
"""

# pip install scikit-learn===1.1.3

#%%

# 회귀(Regression) - 보스턴 주택 가격 예측
# 적용모델
#   - 베이스라인 모델 - 선형회귀
#   - 과대적합 회피(L2/L1 규제)
#   - Ridge(L2 규제) 모델
#   - Lasso(L1 규제) 모델
#   - ElasticNet(L1, L2 규제) 모델
#   - 의사결정나무(DecisionTreeRegressor)
#   - 랜덤포레스트(RandomForestRegressor)
#   - 부스팅(XGBRegressor)

#%%
# # 라이브러리 설정

# # 데이터셋 불러오기

# In[1]:


# 기본 라이브러리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# from sklearn import datasets

#%%

# skleran 데이터셋에서 보스턴 주택 데이터셋 로딩
# housing = datasets.load_boston()

# CSV 파일에서 데이터셋 로딩
data = pd.read_csv("./boston_housing.csv")
target = pd.read_csv("./boston_housing_target.csv")

# 딕셔너리 형태이므로, key 값을 확인
# housing.keys()


# In[2]:

# 판다스 데이터프레임으로 변환
# data = pd.DataFrame(housing['data'], columns=housing['feature_names'])
# target = pd.DataFrame(housing['target'], columns=['Target'])
# 데이터셋 크기
print(data.shape)
print(target.shape)


#%%
# CSV로 저장
# 인덱스는 제외
"""
data.to_csv("./boston_housing.csv", index=False)
target.to_csv("./boston_housing_target.csv", index=False)
"""

# 다시 읽기
"""
data1 = pd.read_csv("./boston_housing.csv")
target1 = pd.read_csv("./boston_housing_target.csv")
"""

# In[3]:


# 데이터프레임 결합하기
df = pd.concat([data, target], axis=1)
df.head(2)


# # 데이터 탐색 (EDA)

# In[4]:


# 데이터프레임의 기본정보
df.info()


# In[5]:


# 결측값 확인
# 결과 : 없음
df.isnull().sum()


# In[6]:

# 상관계수 행렬
df_corr = df.corr()

# 히트맵 그리기
plt.figure(figsize=(10, 10))
sns.set(font_scale=0.8)
sns.heatmap(df_corr, annot=True, cbar=False);
plt.show()


# In[7]:


# 변수 간의 상관관계 분석 - Target 변수와 상관관계가 높은 순서대로 정리
# Target의 절대값을 기준으로 내림차순 정렬
corr_order = df_corr.loc[:'LSTAT', 'Target'].abs().sort_values(ascending=False)
corr_order

#%%

"""
LSTAT      0.737663  # 저소득층 비율
RM         0.695360  # 방의 갯수
PTRATIO    0.507787  # 교사-학생 비율
INDUS      0.483725  # 해당 지역의 비소매 상버 지역 비율
TAX        0.468536  # 재산세
NOX        0.427321
CRIM       0.388305
RAD        0.381626
AGE        0.376955
ZN         0.360445
B          0.333461
DIS        0.249929
CHAS       0.175260
Name: Target, dtype: float64
"""

# In[8]:


# Target 변수와 상관관계가 높은 4개 변수를 추출
plot_cols = ['Target', 'LSTAT', 'RM', 'PTRATIO', 'INDUS']
plot_df = df.loc[:, plot_cols]
plot_df.head()

#%%

"""
   Target  LSTAT     RM  PTRATIO  INDUS
0    24.0   4.98  6.575     15.3   2.31
1    21.6   9.14  6.421     17.8   7.07
2    34.7   4.03  7.185     17.8   7.07
3    33.4   2.94  6.998     18.7   2.18
4    36.2   5.33  7.147     18.7   2.18
"""

# In[9]:

# regplot으로 선형회귀선 표시
plt.figure(figsize=(10,10))
for idx, col in enumerate(plot_cols[1:]):
    ax1 = plt.subplot(2, 2, idx+1) # 2행 2열의 그래프
    sns.regplot(x=col, y=plot_cols[0], data=plot_df, ax=ax1)    
plt.show()


# In[10]:


# Target 데이터의 분포
sns.displot(x='Target', kind='hist', data=df)
plt.show()


############################################################################
# # 데이터 전처리

# ### 피처 스케일링

# In[11]:


# 사이킷런 MinMaxScaler 적용 
# 정규화(Normalization) : 값의 크기를 비슷한 수준으로 조정
# 0~1사이 값으로 변환
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

df_scaled = df.iloc[:, :-1] # Target 데이터 제외

#%%

# 결과: Array of float64
scaler.fit(df_scaled)
df_scaled = scaler.transform(df_scaled)

#%%

# 스케일링 된 값으로 교체
# 스케일링 변환된 값을 데이터프레임에 반영
# df.iloc[:, :-1] = df_scaled[:, :]
# df.head()

ndf = df.copy()
ndf.iloc[:,:-1] = df_scaled[:, :]
ndf.head()


#%%
# ### 학습용-테스트 데이터셋 분리하기

# In[12]:


# 학습 - 테스트 데이터셋 분할
from sklearn.model_selection import train_test_split
X_data = ndf.loc[:, ['LSTAT', 'RM']] # 저소득층비율, 거주목적의 방의 갯수
y_data = ndf.loc[:, 'Target'] # 정답
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, 
                                                    test_size=0.2, # 테스트 데이터 20%
                                                    shuffle=True,  # 데이터를 섞음
                                                    random_state=12)
print(X_train.shape, y_train.shape) # (404, 2) (404,)
print(X_test.shape, y_test.shape)   # (102, 2) (102,)


#%%
## Baseline 모델 - 선형 회귀

# 선형 회귀 모형
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

print ("회귀계수(기울기): ", np.round(lr.coef_, 1))  # [-23.2  25.4]
print ("상수항(절편): ", np.round(lr.intercept_, 1)) # 16.3


# In[14]:


# 예측
y_test_pred = lr.predict(X_test)

# 예측값, 실제값의 분포
plt.figure(figsize=(10, 5))
plt.scatter(X_test['LSTAT'], y_test, label='y_test')              # 정답
plt.scatter(X_test['LSTAT'], y_test_pred, c='r', label='y_pred')  # 예측(빨강)
plt.legend(loc='best')
plt.show()


# In[15]:


# 평가
from sklearn.metrics import mean_squared_error
y_train_pred = lr.predict(X_train) # 훈련 데이터로 예측

# 훈련데이터의 정답과 훈련데이터로 예측한 결과 비교
train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE: %.4f" % train_mse) # Train MSE: 30.8042

# 테스트데이터의 정답과 테스트데이터로 예측한 결과 비교
test_mse = mean_squared_error(y_test, y_test_pred) # Test MSE: 29.5065
print("Test MSE: %.4f" % test_mse)

#%%
###############################################################################
# ## 교차 검증
###############################################################################

# In[16]:

# cross_val_score 함수
# K-Fold 교차 검증
# 폴드갯수: cv(5)
from sklearn.model_selection import cross_val_score
lr = LinearRegression()
print("mse_scores:", cross_val_score(lr, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
print("mse_scores:", cross_val_score(lr, X_train, y_train, cv=5))

# MSE를 음수로 계산하여 -1을 곱하여 양수로 변환
mse_scores = -1*cross_val_score(lr, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print("개별 Fold의 MSE: ", np.round(mse_scores, 4)) # [31.465  34.668  28.9147 29.3535 34.6627]
print("평균 MSE: %.4f" % np.mean(mse_scores))       #  31.8128


#%%

###############################################################################
# 과대적합 회피
# # L1/L2 규제

# L1: 가중치 절대값의 합에 패널티를 부여하여 모델의 복잡도를 낮춤
#  - Lasso 모델

# L2: 가중치 제곱합에 패널티를 부여하여 모델의 복잡도를 낮춤
#  - Ridge 모델

# In[17]:


# 2차 다항식 변환
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2) # 2차
X_train_poly = pf.fit_transform(X_train)
print("원본 학습 데이터셋: ", X_train.shape)
print("2차 다항식 변환 데이터셋: ", X_train_poly.shape)


# In[18]:


# 2차 다항식 변환 데이터셋으로 선형 회귀 모형 학습
lr = LinearRegression()
lr.fit(X_train_poly, y_train)

#%%
# 테스트 데이터에 대한 예측 및 평가
y_train_pred = lr.predict(X_train_poly)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE: %.4f" % train_mse) # Train MSE: 21.5463

X_test_poly = pf.fit_transform(X_test)
y_test_pred = lr.predict(X_test_poly)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE: %.4f" % test_mse) # Test MSE: 16.7954


# In[19]:

print("# 15차 다항식 변환 데이터셋으로 선형 회귀 모형 학습")

# 15차 다항식 변환 데이터셋으로 선형 회귀 모형 학습
pf = PolynomialFeatures(degree=15)
X_train_poly = pf.fit_transform(X_train)

lr = LinearRegression()
lr.fit(X_train_poly, y_train)

# 테스트 데이터에 대한 예측 및 평가
y_train_pred = lr.predict(X_train_poly)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE: %.4f" % train_mse) # Train MSE: 11.2109

X_test_poly = pf.fit_transform(X_test)
y_test_pred = lr.predict(X_test_poly)
test_mse = mean_squared_error(y_test, y_test_pred)

# 과대적합 : 예측력을 상실
print("Test MSE: %.4f" % test_mse) # Test MSE: 95441494600181.1250


# In[27]:


# 다항식 차수에 따른 모델 적합도 변화
plt.figure(figsize=(15,5))
for n, deg in enumerate([1, 2, 15]): # 1차, 2차, 15차
    ax1 = plt.subplot(1, 3, n+1)
    # degree별 다항 회귀 모형 적용
    pf = PolynomialFeatures(degree=deg)
    X_train_poly = pf.fit_transform(X_train.loc[:, ['LSTAT']])
    X_test_poly = pf.fit_transform(X_test.loc[:, ['LSTAT']])
    lr = LinearRegression()
    lr.fit(X_train_poly, y_train)
    y_test_pred = lr.predict(X_test_poly)
    # 실제값 분포
    plt.scatter(X_test.loc[:, ['LSTAT']], y_test, label='Targets') 
    # 예측값 분포
    plt.scatter(X_test.loc[:, ['LSTAT']], y_test_pred, label='Predictions') 
    # 제목 표시
    plt.title("Degree %d" % deg)
    # 범례 표시
    plt.legend()  
plt.show()

#%%

# 결과: 데이터가 곡선 형태
#   - 1차: 직선형태로 부족하다
#   - 2차: 설명력이 좋아졌다.
#   - 15차: 회귀곡선의 변곡점이 많아져 불안정

# In[21]:

  
###############################################################################
# Ridge (L2 규제)
# L2: 가중치 제곱합에 패널티를 부여하여 모델의 복잡도를 낮춤
# alpha: 값이 증가하면 규제강도가 커지고 모델의 가중치를 감소시킴
# 학습데이터와 훈련데이터의 차이
from sklearn.linear_model import Ridge
rdg = Ridge(alpha=2.5)
rdg.fit(X_train_poly, y_train)

y_train_pred = rdg.predict(X_train_poly)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE: %.4f" % train_mse) # 35.9484

y_test_pred = rdg.predict(X_test_poly)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE: %.4f" % test_mse) # 42.0011


# In[22]:

# Lasso (L1 규제)
# L1: 가중치 절대값의 합에 패널티를 부여하여 모델의 복잡도를 낮춤
from sklearn.linear_model import Lasso
las = Lasso(alpha=0.05)
las.fit(X_train_poly, y_train)

y_train_pred = las.predict(X_train_poly)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE: %.4f" % train_mse)   # 32.3204

y_test_pred = las.predict(X_test_poly)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE: %.4f" % test_mse)     # 37.7103


# In[23]:

# ElasticNet (L2/L1 규제)
# L1/L2 규제 모드 적용한 선형 회귀모델
# alpha : L2와 L1 규제 강도의 합
# l1_ratio: L1 규제 강도의 상대적 비율 조정
#   - 0 : L2 규제와 같다.
#   - 1 : L2 규제와 같다.
from sklearn.linear_model import ElasticNet
ela = ElasticNet(alpha=0.01, l1_ratio=0.7)
ela.fit(X_train_poly, y_train)

y_train_pred = ela.predict(X_train_poly)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE: %.4f" % train_mse)  # 33.7551

y_test_pred = ela.predict(X_test_poly)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE: %.4f" % test_mse)    # 39.4968


#%%
# # 트리 기반 모델 - 비선형 회귀

# 의사결정 나무
# 학습과 검증의 차이가 크기 않다.
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(max_depth=3, random_state=12)
dtr.fit(X_train, y_train)

y_train_pred = dtr.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE: %.4f" % train_mse)   # 18.8029

y_test_pred = dtr.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE: %.4f" % test_mse)     # 17.9065


# In[25]:

# 랜덤 포레스트
# 의사결정나무에 비해서 예측력이 개선
# 약간의 과대적합된 경이 있다. (훈련보다 검증이 높다)
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(max_depth=3, random_state=12)
rfr.fit(X_train, y_train)

y_train_pred = rfr.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE: %.4f" % train_mse)  # 16.0201

y_test_pred = rfr.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE: %.4f" % test_mse)    # 17.7751


# In[26]:

# XGBoost
# 랜덤포레스트와 비교해서 예측력이 향상된 것 처럼 보이지만
# 훈련과 검증의 차이가 커서 과대적합이 심화
# XGBoost 특징:
#   - 복잡도가 높은 알고리즘으로 데이터의 갯수가 작으면 과대적합 될 위험성 높다.
#   - 데이터의 갯수가 많고, 모델 예측의 난이도가 높은 경우 탁월한 성능을 발휘한다.    
from xgboost import XGBRegressor
xgbr = XGBRegressor(objective='reg:squarederror', max_depth=3, random_state=12)
xgbr.fit(X_train, y_train)

y_train_pred = xgbr.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE: %.4f" % train_mse)  # 3.9261

y_test_pred = xgbr.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE: %.4f" % test_mse)    # 19.9509