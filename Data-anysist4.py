# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:59:12 2024

@author: leehj
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

df = pd.read_csv('./data/part5/auto-mpg.csv', header=None)

df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']

print(df.head()) 
print('\n') 

pd.set_option('display.max_columns', 10)
print(df.head())    # 5행 출력 

"""
 mpg  cylinders  displacement  ... model year  origin                       name
0  18.0          8         307.0  ...         70       1  chevrolet chevelle malibu
1  15.0          8         350.0  ...         70       1          buick skylark 320
2  18.0          8         318.0  ...         70       1         plymouth satellite
3  16.0          8         304.0  ...         70       1              amc rebel sst
4  17.0          8         302.0  ...         70       1                ford torino

[5 rows x 9 columns]


    mpg  cylinders  displacement horsepower  weight  acceleration  model year  \
0  18.0          8         307.0      130.0  3504.0          12.0          70   
1  15.0          8         350.0      165.0  3693.0          11.5          70   
2  18.0          8         318.0      150.0  3436.0          11.0          70   
3  16.0          8         304.0      150.0  3433.0          12.0          70   
4  17.0          8         302.0      140.0  3449.0          10.5          70   

   origin                       name  
0       1  chevrolet chevelle malibu  
1       1          buick skylark 320  
2       1         plymouth satellite  
3       1              amc rebel sst  
4       1                ford torino  
"""

#%% 

print(df.info()) 
print('\n') 
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 398 entries, 0 to 397
Data columns (total 9 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   mpg           398 non-null    float64
 1   cylinders     398 non-null    int64  
 2   displacement  398 non-null    float64
 3   horsepower    398 non-null    object 
 4   weight        398 non-null    float64
 5   acceleration  398 non-null    float64
 6   model year    398 non-null    int64  
 7   origin        398 non-null    int64  
 8   name          398 non-null    object 
dtypes: float64(4), int64(3), object(2)
memory usage: 28.1+ KB
None
"""

#%%
# 데잍 통계 요약 정보 확인 
print(df.describe()) 

"""
          mpg   cylinders  displacement       weight      acceleration  
count  398.000000  398.000000    398.000000   398.000000    398.000000   
mean    23.514573    5.454774    193.425879  2970.424623     15.568090   
std      7.815984    1.701004    104.269838   846.841774      2.757689   
min      9.000000    3.000000     68.000000  1613.000000      8.000000   
25%     17.500000    4.000000    104.250000  2223.750000     13.825000   
50%     23.000000    4.000000    148.500000  2803.500000     15.500000   
75%     29.000000    8.000000    262.000000  3608.000000     17.175000   
max     46.600000    8.000000    455.000000  5140.000000     24.800000   

       model year      origin  
count  398.000000  398.000000  
mean    76.010050    1.572864  
std      3.697627    0.802055  
min     70.000000    1.000000  
25%     73.000000    1.000000  
50%     76.000000    1.000000  
75%     79.000000    2.000000  
max     82.000000    3.000000  
"""

#%%
print(df['horsepower'].unique()) 
print('\n') 

df['horsepower'].replace('?', np.nan, inplace=True)     # '?'을 np.nan으로 변경 
df.dropna(subset=['horsepower'], axis=0, inplace=True)  # 누락 데이터 행 삭제  
df['horsepower'] = df['horsepower'].astype('float')     # 문자열을 실수형으로 변환 

print(df.describe()) 

"""
          mpg   cylinders  displacement  horsepower       weight  \
count  392.000000  392.000000    392.000000  392.000000   392.000000   
mean    23.445918    5.471939    194.411990  104.469388  2977.584184   
std      7.805007    1.705783    104.644004   38.491160   849.402560   
min      9.000000    3.000000     68.000000   46.000000  1613.000000   
25%     17.000000    4.000000    105.000000   75.000000  2225.250000   
50%     22.750000    4.000000    151.000000   93.500000  2803.500000   
75%     29.000000    8.000000    275.750000  126.000000  3614.750000   
max     46.600000    8.000000    455.000000  230.000000  5140.000000   

       acceleration  model year      origin  
count    392.000000  392.000000  392.000000  
mean      15.541327   75.979592    1.576531  
std        2.758864    3.683737    0.805518  
min        8.000000   70.000000    1.000000  
25%       13.775000   73.000000    1.000000  
50%       15.500000   76.000000    1.000000  
75%       17.025000   79.000000    2.000000  
max       24.800000   82.000000    3.000000
"""

#%% 
ndf = df[['mpg', 'cylinders','horsepower','weight']]
print(ndf.head()) 
# 속성 select 
"""
mpg  cylinders  horsepower  weight
0  18.0          8       130.0  3504.0
1  15.0          8       165.0  3693.0
2  18.0          8       150.0  3436.0
3  16.0          8       150.0  3433.0
4  17.0          8       140.0  3449.0
"""
### 중속변수(X), 독립 변수(X) 두 변수간에 선형관계에 그래프를 그려서 확인 
# scatter 함수 --> 산점도 측정 

ndf.plot(kind='scatter', x='weight', y='mpg', c='coral', s=10, figsize=(10,5)) 
plt.show()
plt.close() 

#%% 
# 단순회귀 분석 
fig = plt.figure(figsize=(10,5)) 
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2) 
sns.regplot(x='weight', y='mpg', data=ndf, ax=ax1)
sns.regplot(x='weight', y='mpg', data=ndf, ax=ax2, fit_reg=False)  # fit_reg=False => 회귀선 제거 옵션 
plt.show()
plt.close() 

#%% 
# joinplot() 함수도 사용 가능 --> 2개 열 
sns.jointplot(x='weight', y='mpg', data=ndf)
sns.jointplot(x='weight', y='mpg', kind='reg', data=ndf) 
plt.show()
plt.close() 

#%%
# pairplot() 함수 --> 여러개 모든 경우의 수 표출 
grid_ndf = sns.pairplot(ndf)
plt.show()
plt.close() 
#%% 
# 훈련/검증 데이터 분할 
# 독립변수 horsepower / weight 열을 독립 변수 x로 선택 고려 
# 훈련set, 검증set 모형 구축 
# 속성 변수 선택 
x = ndf[['weight']]
y = ndf[['mpg']]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=10) 

print('train data 개수: ', len(x_train))
print('test data 개수:', len(x_test)) 
"""
train data 개수:  274
test data 개수: 118
"""

#%% 
# sklearn 라이브러리 에서 선형회귀 분석 모듈 사용 
from sklearn.linear_model import LinearRegression 

lr = LinearRegression() 

lr.fit(x_train, y_train)


# 학습을 마친 모형에 test data를 적용하여 결정계수(R-제곱) 계산 
r_square = lr.score(x_test, y_test)
print(r_square) 
# 0.6822458558299325 
#%% 
print('기울기 a: ', lr.coef_)     # a 회귀식 기울기 
print('\n')  

print('y절편 b', lr.intercept_)  # y 절편에 속성값 

"""
기울기 a:  [[-0.00775343]]


y절편 b [46.71036626]
"""

y_hat = lr.predict(x)                    # # 예측값 y_hat, 실제 값 y와 비교 

plt.figure(figsize=(10, 5)) 
ax1 = sns.kdeplot(y, label="y")
ax2 = sns.kdeplot(y_hat, label="y_hat", ax=ax1)
plt.legend()
plt.show() 














#%% 

# 다중 회귀 분석 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 



df = pd.read_csv('./data/part5/auto-mpg.csv', header=None) 

df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']

# horsepower 열의 자료형 변경(문자열->숫자) 
df['horsepower'].replace('?', np.nan, inplace=True) 
df.dropna(subset=['horsepower'], axis=0, inplace=True)
df['horsepower'] = df['horsepower'].astype('float') 

# 분석에 활용한 열 
ndf = df[['mpg', 'cylinders', 'horsepower', 'weight']]

x = ndf[['weight']] # 독립변수 x
y = ndf[['mpg']]    # 종속변수 y 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=10) 

print('훈련 데이터:', x_train.shape)
print('검증 데이터:', x_test.shape) 

"""
훈련 데이터: (274, 1)
검증 데이터: (118, 1)
"""

#%% 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
x_train_poly=poly.fit_transform(x_train) 

print('원 데이터:', x_train.shape)
print('2자항 변환 데이터:', x_train_poly.shape) 

"""
원 데이터: (274, 1)
2자항 변환 데이터: (274, 3)
"""

#%% 
pr = LinearRegression()
pr.fit(x_train_poly, y_train)

x_test_poly = poly.fit_transform(x_test)
r_square = pr.score(x_test_poly,y_test) 
print(r_square) 


# 0.7087009262975481

#%% 
# sklearn 라이브러리에서 필요한 모듈 가져오기 
y_hat_test = pr.predict(x_test_poly)

fig = plt.figure(figsize=(10,5)) 
ax = fig.add_subplot(1,1,1) 
ax.plot(x_train, y_train,'o', label='Train Data') 
ax.plot(x_test, y_hat_test, 'r+', label='Predicted Value') 
ax.legend(loc='best')
plt.xlabel('weight')
plt.ylabel('mpg')
plt.show()
plt.close() 

#%% 
# 예측값 y_hat, 실제 값 y와 비교 
x_ploy = poly.fit_transform(x) 
y_hat = pr.predict(x_ploy)  

plt.figure(figsize=(10,5))
ax1 = sns.kdeplot(y, label="y") 
ax2 = sns.kdeplot(y_hat, label="y_hat", ax=ax1)
plt.legend()
plt.show() 

#%%
# 다중회귀 분석 알고리즘 : 여러개의 독립변수가 종속 변수에 영향을 주고 선형 관계를 갖는 경우 
# 각 독립 변수의 계수와 상수항에 적절한 값들을 찾아서 모형을 완성 
# 지도학습 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 


df = pd.read_csv('./data/part5/auto-mpg.csv', header=None) 

df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']

df['horsepower'].replace('?',np.nan, inplace=True)
df.dropna(subset=['horsepower'], axis=0, inplace=True) 
df['horsepower'] = df['horsepower'].astype('float') 

# 분석에 활용할 열 

ndf = df[['mpg','cylinders','horsepower','weight']]

x = ndf[['cylinders','horsepower','weight']]
y = ndf['mpg']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10) 

print('훈련 데이터: ',x_train.shape)
print('검증 데이터:', x_test.shape) 

"""
훈련 데이터:  (274, 3)
검증 데이터: (118, 3)
"""
#%% 

from sklearn.linear_model import LinearRegression

lr = LinearRegression() 

lr.fit(x_train, y_train)

r_square = lr.score(x_test, y_test)
print(r_square)
print('\n')

print('x 변수의 계수 a:',lr.coef_)
print('\n')

print('상수항 b', lr.intercept_) 

"""
0.6939048496695599

x 변수의 계수 a: [-0.60691288 -0.03714088 -0.00522268]

상수항 b 46.414351269634025
"""


#%% 

y_hat = lr.predict(x_test)

plt.figure(figsize=(10,5))
ax1 = sns.kdeplot(y_test, label="y_test")
ax2 = sns.kdeplot(y_hat, label="y_hat",ax=ax1)
plt.legend()
plt.show() 





import pandas as pd
import seaborn as sns


# load_dataset 함수를 사용하여 데이터프레임으로 변환 
df = sns.load_dataset('titanic')

print(df.head()) 
"""
urvived  pclass     sex   age  ...  deck  embark_town  alive  alone
0         0       3    male  22.0  ...   NaN  Southampton     no  False
1         1       1  female  38.0  ...     C    Cherbourg    yes  False
2         1       3  female  26.0  ...   NaN  Southampton    yes   True
3         1       1  female  35.0  ...     C  Southampton    yes  False
4         0       3    male  35.0  ...   NaN  Southampton     no   True
"""

print('\n')


# 열 개수 
pd.set_option('display.max_columns',15)
print(df.head()) 
"""
survived  pclass     sex   age  ...  deck  embark_town  alive  alone
0         0       3    male  22.0  ...   NaN  Southampton     no  False
1         1       1  female  38.0  ...     C    Cherbourg    yes  False
2         1       3  female  26.0  ...   NaN  Southampton    yes   True
3         1       1  female  35.0  ...     C  Southampton    yes  False
4         0       3    male  35.0  ...   NaN  Southampton     no   True

[5 rows x 15 columns]


   survived  pclass     sex   age  sibsp  parch     fare embarked  class  \
0         0       3    male  22.0      1      0   7.2500        S  Third   
1         1       1  female  38.0      1      0  71.2833        C  First   
2         1       3  female  26.0      0      0   7.9250        S  Third   
3         1       1  female  35.0      1      0  53.1000        S  First   
4         0       3    male  35.0      0      0   8.0500        S  Third   

     who  adult_male deck  embark_town alive  alone  
0    man        True  NaN  Southampton    no  False  
1  woman       False    C    Cherbourg   yes  False  
2  woman       False  NaN  Southampton   yes   True  
3  woman       False    C  Southampton   yes  False  
4    man        True  NaN  Southampton    no   True  
"""
#%% 
# 분류 
# KNN 분류 알고리즘 
# 새로운 관측값 이 주어지면 기존 데이터 중에서 가장 속성이 같은 k개 이웃을 찾음
# 가까운 이웃들이 갖고 있는 목표 값, 같은 값으로 분류하여 예측 

import pandas as pd
import seaborn as sns

'''
[Step 1] 데이터 준비 - Seaborn에서 제공하는 titanic 데이터셋 가져오기
'''

df = sns.load_dataset('titanic')   # load_dataset 함수 

print(df.head())   # 데이터in 
print('\n')

#  IPython 디스플레이 설정 - 출력할 열의 개수 한도 늘리기
pd.set_option('display.max_columns', 15)
print(df.head())   
print('\n')


'''
[Step 2] 데이터 탐색
'''

# 데이터 자료형 확인
print(df.info())                 # deck 유효값 203개 --> 전제 891명중 688명의 데이터 존제 x-> 제거 
print('\n')                      # embarked 열, embark_town 동일의미 -> 제거 

"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 15 columns):
 #   Column       Non-Null Count  Dtype   
---  ------       --------------  -----   
 0   survived     891 non-null    int64   
 1   pclass       891 non-null    int64   
 2   sex          891 non-null    object  
 3   age          714 non-null    float64 
 4   sibsp        891 non-null    int64   
 5   parch        891 non-null    int64   
 6   fare         891 non-null    float64 
 7   embarked     889 non-null    object  
 8   class        891 non-null    category
 9   who          891 non-null    object  
 10  adult_male   891 non-null    bool    
 11  deck         203 non-null    category
 12  embark_town  889 non-null    object  
 13  alive        891 non-null    object  
 14  alone        891 non-null    bool    
"""


rdf = df.drop(['deck','embark_town'], axis=1)   # drop 사용 열제거 
print(rdf.columns.values) 

"""

['survived' 'pclass' 'sex' 'age' 'sibsp' 'parch' 'fare' 'embarked' 'class'
 'who' 'adult_male' 'alive' 'alone']

"""

#%% 
# 나이 데이터가 없는 행 모두 삭제 - age 열(891개 중 177개 NaN값)
rdf = rdf.dropna(subset=['age'], how='any', axis=0)    # 714개 데이터 승객으로 분석 
print(len(rdf)) 
"""
714
"""
#%% 
# id max값을 사용하여 embarked 열의 NaN값을 승선도시 중에서 가장 많이 출현한 값으로 치환 --> value_counts 
most_freq = rdf['embarked'].value_counts(dropna=True).idxmax() 
print(most_freq)
print('\n')

print(rdf.describe(include='all'))
print('\n')

rdf['embarked'].fillna(most_freq, inplace=True) 

"""
survived      pclass   sex         age       sibsp       parch  \
count   714.000000  714.000000   714  714.000000  714.000000  714.000000   
unique         NaN         NaN     2         NaN         NaN         NaN   
top            NaN         NaN  male         NaN         NaN         NaN   
freq           NaN         NaN   453         NaN         NaN         NaN   
mean      0.406162    2.236695   NaN   29.699118    0.512605    0.431373   
std       0.491460    0.838250   NaN   14.526497    0.929783    0.853289   
min       0.000000    1.000000   NaN    0.420000    0.000000    0.000000   
25%       0.000000    1.000000   NaN   20.125000    0.000000    0.000000   
50%       0.000000    2.000000   NaN   28.000000    0.000000    0.000000   
75%       1.000000    3.000000   NaN   38.000000    1.000000    1.000000   
max       1.000000    3.000000   NaN   80.000000    5.000000    6.000000   

              fare embarked  class  who adult_male alive alone  
count   714.000000      714    714  714        714   714   714  
unique         NaN        3      3    3          2     2     2  
top            NaN        S  Third  man       True    no  True  
freq           NaN      556    355  413        413   424   404  
mean     34.694514      NaN    NaN  NaN        NaN   NaN   NaN  
std      52.918930      NaN    NaN  NaN        NaN   NaN   NaN  
min       0.000000      NaN    NaN  NaN        NaN   NaN   NaN  
25%       8.050000      NaN    NaN  NaN        NaN   NaN   NaN  
50%      15.741700      NaN    NaN  NaN        NaN   NaN   NaN  
75%      33.375000      NaN    NaN  NaN        NaN   NaN   NaN  
max     512.329200      NaN    NaN  NaN        NaN   NaN   NaN  
"""
#%% 
# 속성 선택

ndf = rdf[['survived','pclass','sex','age','sibsp','parch','embarked']]
print(ndf.head()) 

"""
survived  pclass  sex   age  sibsp  parch embarked
0         0       3    male  22.0      1      0        S
1         1       1  female  38.0      1      0        C
2         1       3  female  26.0      0      0        S
3         1       1  female  35.0      1      0        S
4         0       3    male  35.0      0      0        S
"""

#%%

onehot_sex = pd.get_dummies(ndf['sex'])
ndf = pd.concat([ndf, onehot_sex], axis=1) 

onehot_embarked = pd.get_dummies(ndf['embarked'], prefix='town') 
ndf = pd.concat([ndf, onehot_embarked], axis=1)

ndf.drop(['sex','embarked'], axis=1, inplace=True)
print(ndf.head()) 
"""
      survived  pclass  sex   age      sibsp  parch embarked
0         0       3    male  22.0      1      0        S
1         1       1  female  38.0      1      0        C
2         1       3  female  26.0      0      0        S
3         1       1  female  35.0      1      0        S
4         0       3    male  35.0      0      0        S
   survived  pclass   age  sibsp  parch  female   male  town_C  town_Q  town_S
0         0       3  22.0      1      0   False   True   False   False    True
1         1       1  38.0      1      0    True  False    True   False   False
2         1       3  26.0      0      0    True  False   False   False    True
3         1       1  35.0      1      0    True  False   False   False    True
4         0       3  35.0      0      0   False   True   False   False    True
"""

x = ndf[['pclass', 'age', 'sibsp','parch','female','male',
         'town_C','town_Q','town_S']]

y = ndf['survived'] 

#%% 
from sklearn import preprocessing
x = preprocessing.StandardScaler().fit(x).transform(x)  

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10) 

print('train data 개수: ', x_train.shape)
print('test data 개수:' , x_test.shape) 

"""
train data 개수:  (499, 9)
test data 개수: (215, 9)
"""
#%% 
from sklearn.neighbors import KNeighborsClassifier 

knn = KNeighborsClassifier(n_neighbors=5) 

knn.fit(x_train, y_train)

y_hat = knn.predict(x_test)

print(y_hat[0:10])
print(y_test.values[0:10]) 

"""
[0 0 1 0 0 1 1 1 0 0]
[0 0 1 0 0 1 1 1 0 0]
"""
#%%
from sklearn import metrics
knn_matrix = metrics.confusion_matrix(y_test, y_hat)
print(knn_matrix) 

"""
[[110  15]
 [ 26  64]]
"""
#%% 
knn_report = metrics.classification_report(y_test, y_hat)
print(knn_report) 

"""
[0 0 1 0 0 1 1 1 0 0]
[0 0 1 0 0 1 1 1 0 0]
[[110  15]
 [ 26  64]]
              precision    recall  f1-score   support

           0       0.81      0.88      0.84       125
           1       0.81      0.71      0.76        90

    accuracy                           0.81       215
   macro avg       0.81      0.80      0.80       215
weighted avg       0.81      0.81      0.81       215
"""













#%%
# SVM 알고리즘 
# SVM : 학습을 통해 백터 공간을 나누는 경계를 찾는다. 
import pandas as pd
import seaborn as sns




df = sns.load_dataset('titanic')


pd.set_option('display.max_columns', 15)


rdf = df.drop(['deck', 'embark_town'], axis=1)  
rdf = rdf.dropna(subset=['age'], how='any', axis=0)  


most_freq = rdf['embarked'].value_counts(dropna=True).idxmax()   
rdf['embarked'].fillna(most_freq, inplace=True)




# 분석에 활용할 열(속성)을 선택 
ndf = rdf[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'embarked']]

# 원핫인코딩 - 범주형 데이터를 모형이 인식할 수 있도록 숫자형으로 변환
onehot_sex = pd.get_dummies(ndf['sex'])
ndf = pd.concat([ndf, onehot_sex], axis=1)

onehot_embarked = pd.get_dummies(ndf['embarked'], prefix='town')
ndf = pd.concat([ndf, onehot_embarked], axis=1)

ndf.drop(['sex', 'embarked'], axis=1, inplace=True)






X=ndf[['pclass', 'age', 'sibsp', 'parch', 'female', 'male', 
       'town_C', 'town_Q', 'town_S']]  #독립 변수 X
y=ndf['survived']                      #종속 변수 Y





# 설명 변수 데이터를 정규화(normalization)
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10) 

print('train data 개수: ', X_train.shape)
print('test data 개수: ', X_test.shape)
print('\n')


"""
train data 개수:  (499, 9)
test data 개수:  (215, 9)
"""



#%% 


# sklearn 라이브러리에서 SVM 분류 모형 가져오기
from sklearn import svm

# 모형 객체 생성 (kernel='rbf' 적용)
svm_model = svm.SVC(kernel='rbf')


svm_model.fit(X_train, y_train)   

# test data를 가지고 y_hat을 예측 (분류) 
y_hat = svm_model.predict(X_test)

print(y_hat[0:10])
print(y_test.values[0:10])
print('\n')

# 모형 성능 평가 - Confusion Matrix 계산: 미생존자 0, 생존자 1의 값 
from sklearn import metrics 
svm_matrix = metrics.confusion_matrix(y_test, y_hat)  
print(svm_matrix)
print('\n')

# 모형 성능 평가 - 평가지표 계산
svm_report = metrics.classification_report(y_test, y_hat)            
print(svm_report)


"""
[[120   5]    -> 215명중 TN값 120명, FP값 5명
 [ 35  55]]   -> FN값 35명, TP값 55명    --> 미생존자 정확도 예측값: 0.86, 생존자 정확도: 0.73 


              precision    recall  f1-score   support

           0       0.77      0.96      0.86       125
           1       0.92      0.61      0.73        90

    accuracy                           0.81       215
   macro avg       0.85      0.79      0.80       215
weighted avg       0.83      0.81      0.81       215



"""


