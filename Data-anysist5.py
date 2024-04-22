# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:09:51 2024

@author: leehj
"""

import pandas as pd
import numpy as np 


# Decision tree  # 각 분기점마다 목표 값을 가장 잘 분류할 수 있는 속성을 배치 
# 해당 속성이 갖는 값을 이용해 새로운 가지를 만든다 
# 속성 기준으로 분류한 값들이 구분 되는 정도를 측정 
# 다른 종류의 값들이 섞여 있는 정도 Entropy를 주로 활용 
# Entropy 가 낮을 수록 분류 check max 
uci_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/\
breast-cancer-wisconsin/breast-cancer-wisconsin.data'

df = pd.read_csv(uci_path, header=None)

df.columns = ['id','clump','cell_size','cell_shape','adhesion','epithlial',
              'bare_nuclei','chromatin','normal_nucleoli','mitoses','class']

pd.set_option('display.max_columns', 15) 

"""
id       clump   cell_size  cell_shape    adhesion  \
count  6.990000e+02  699.000000  699.000000  699.000000  699.000000   
mean   1.071704e+06    4.417740    3.134478    3.207439    2.806867   
std    6.170957e+05    2.815741    3.051459    2.971913    2.855379   
min    6.163400e+04    1.000000    1.000000    1.000000    1.000000   
25%    8.706885e+05    2.000000    1.000000    1.000000    1.000000   
50%    1.171710e+06    4.000000    1.000000    1.000000    1.000000   
75%    1.238298e+06    6.000000    5.000000    5.000000    4.000000   
max    1.345435e+07   10.000000   10.000000   10.000000   10.000000   

epithlial   chromatin  normal_nucleoli     mitoses       class  
count  699.000000  699.000000       699.000000  699.000000  699.000000  
mean     3.216023    3.437768         2.866953    1.589413    2.689557  
std      2.214300    2.438364         3.053634    1.715078    0.951273  
min      1.000000    1.000000         1.000000    1.000000    2.000000  
25%      2.000000    2.000000         1.000000    1.000000    2.000000  
50%      2.000000    3.000000         1.000000    1.000000    2.000000  
75%      4.000000    5.000000         4.000000    1.000000    4.000000  
max     10.000000   10.000000        10.000000   10.000000    4.000000  
"""



#%%

print(df.head())
print('\n')

"""
   id  clump  cell_size  cell_shape  adhesion  epithlial bare_unclei  \
0  1000025      5          1           1         1          2           1   
1  1002945      5          4           4         5          7          10   
2  1015425      3          1           1         1          2           2   
3  1016277      6          8           8         1          3           4   
4  1017023      4          1           1         3          2           1   

   chromatin  normal_nucleoli  mitoses  class  
0          3                1        1      2  
1          3                2        1      2  
2          3                1        1      2  
3          3                7        1      2  
4          3                1        1      2  
"""



#%%
print(df.info())
print('\n')
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 699 entries, 0 to 698
Data columns (total 11 columns):
 #   Column           Non-Null Count  Dtype 
---  ------           --------------  ----- 
 0   id               699 non-null    int64 
 1   clump            699 non-null    int64 
 2   cell_size        699 non-null    int64 
 3   cell_shape       699 non-null    int64 
 4   adhesion         699 non-null    int64 
 5   epithlial        699 non-null    int64 
 6   bare_unclei      699 non-null    object
 7   chromatin        699 non-null    int64 
 8   normal_nucleoli  699 non-null    int64 
 9   mitoses          699 non-null    int64 
 10  class            699 non-null    int64 
dtypes: int64(10), object(1)
memory usage: 60.2+ KB
None
"""


#%% 
print(df.describe()) 

"""
  id       clump   cell_size  cell_shape    adhesion  \
count  6.990000e+02  699.000000  699.000000  699.000000  699.000000   
mean   1.071704e+06    4.417740    3.134478    3.207439    2.806867   
std    6.170957e+05    2.815741    3.051459    2.971913    2.855379   
min    6.163400e+04    1.000000    1.000000    1.000000    1.000000   
25%    8.706885e+05    2.000000    1.000000    1.000000    1.000000   
50%    1.171710e+06    4.000000    1.000000    1.000000    1.000000   
75%    1.238298e+06    6.000000    5.000000    5.000000    4.000000   
max    1.345435e+07   10.000000   10.000000   10.000000   10.000000   

        epithlial   chromatin  normal_nucleoli     mitoses       class  
count  699.000000  699.000000       699.000000  699.000000  699.000000  
mean     3.216023    3.437768         2.866953    1.589413    2.689557  
std      2.214300    2.438364         3.053634    1.715078    0.951273  
min      1.000000    1.000000         1.000000    1.000000    2.000000  
25%      2.000000    2.000000         1.000000    1.000000    2.000000  
50%      2.000000    3.000000         1.000000    1.000000    2.000000  
75%      4.000000    5.000000         4.000000    1.000000    4.000000  
max     10.000000   10.000000        10.000000   10.000000    4.000000  
"""
#%% 
#6. bare_unclei  699 non-null    object --> 숫자형 데이터 need 
# 699개 행 중에서 16개행 삭제 -->683개의 관측값

print(df['bare_nuclei'].unique()) 
print('\n')

df['bare_nuclei'].replace('?', np.nan, inplace=True)
df.dropna(subset=['bare_nuclei'], axis=0, inplace=True)
df['bare_nuclei'] = df['bare_nuclei'].astype('int')

print(df.describe()) 

"""
Out[12]: '\nid       clump   cell_size  cell_shape    adhesion  count  6.990000e+02  699.000000  699.000000  699.000000  699.000000   \nmean   1.071704e+06    4.417740    3.134478    3.207439    2.806867   \nstd    6.170957e+05    2.815741    3.051459    2.971913    2.855379   \nmin    6.163400e+04    1.000000    1.000000    1.000000    1.000000   \n25%    8.706885e+05    2.000000    1.000000    1.000000    1.000000   \n50%    1.171710e+06    4.000000    1.000000    1.000000    1.000000   \n75%    1.238298e+06    6.000000    5.000000    5.000000    4.000000   \nmax    1.345435e+07   10.000000   10.000000   10.000000   10.000000   \n\nepithlial   chromatin  normal_nucleoli     mitoses       class  \ncount  699.000000  699.000000       699.000000  699.000000  699.000000  \nmean     3.216023    3.437768         2.866953    1.589413    2.689557  \nstd      2.214300    2.438364         3.053634    1.715078    0.951273  \nmin      1.000000    1.000000         1.000000    1.000000    2.000000  \n25%      2.000000    2.000000         1.000000    1.000000    2.000000  \n50%      2.000000    3.000000         1.000000    1.000000    2.000000  \n75%      4.000000    5.000000         4.000000    1.000000    4.000000  \nmax     10.000000   10.000000        10.000000   10.000000    4.000000  \n'

runcell(4, 'E:/Temp/Workspace/Python 머신러닝 판다스 데이터분석/Part별/data-분석5.py')
['1' '10' '2' '4' '3' '9' '7' '?' '5' '8' '6']


                 id       clump   cell_size  cell_shape    adhesion  \
count  6.830000e+02  683.000000  683.000000  683.000000  683.000000   
mean   1.076720e+06    4.442167    3.150805    3.215227    2.830161   
std    6.206440e+05    2.820761    3.065145    2.988581    2.864562   
min    6.337500e+04    1.000000    1.000000    1.000000    1.000000   
25%    8.776170e+05    2.000000    1.000000    1.000000    1.000000   
50%    1.171795e+06    4.000000    1.000000    1.000000    1.000000   
75%    1.238705e+06    6.000000    5.000000    5.000000    4.000000   
max    1.345435e+07   10.000000   10.000000   10.000000   10.000000   

        epithlial  bare_nuclei   chromatin  normal_nucleoli     mitoses  \
count  683.000000   683.000000  683.000000       683.000000  683.000000   
mean     3.234261     3.544656    3.445095         2.869693    1.603221   
std      2.223085     3.643857    2.449697         3.052666    1.732674   
min      1.000000     1.000000    1.000000         1.000000    1.000000   
25%      2.000000     1.000000    2.000000         1.000000    1.000000   
50%      2.000000     1.000000    3.000000         1.000000    1.000000   
75%      4.000000     6.000000    5.000000         4.000000    1.000000   
max     10.000000    10.000000   10.000000        10.000000   10.000000   

            class  
count  683.000000  
mean     2.699854  
std      0.954592  
min      2.000000  
25%      2.000000  
50%      2.000000  
75%      4.000000  
max      4.000000  
"""
#%% 
x = df[['clump','cell_size','cell_shape','adhesion','epithlial',
        'bare_nuclei','chromatin','normal_nucleoli','mitoses']]

y = df['class']  # class 열 선택 (df.columns에서) 


from sklearn import preprocessing
x = preprocessing.StandardScaler().fit(x).transform(x   

from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

print('train data 개수:', x_train.shape)
print('test data 개수:', x_test.shape) 

"""
train data 개수: (478, 9)   # test_size = 0.3 검증 데이터 30% 
test data 개수: (205, 9)    # 훈련 데이터:478개  검증데이터: 205개 
"""

#%%
from sklearn import tree   
# DeciisionTreeClassfiter() 함수 사용하여 모형 객체를 생성 
# 각 분기점에 최적의 속성파악: entropy 값 사용 
 
tree_model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5) 
tree_model.fit(x_train, y_train)

y_hat = tree_model.predict(x_test)    # x_test애 결과값 y_hat에 저장 

print(y_hat[0:10])
print(y_test.values[0:10])  

# y_hat, y_test을 비교하면 10개 데이터 모두 예측값과 실제값과 일치 

"""
[4 4 4 4 4 4 2 2 4 4]
[4 4 4 4 4 4 2 2 4 4]
"""

#%% 
from sklearn import tree

tree_model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5) 

tree_model.fit(x_train, y_train) 

y_hat = tree_model.predict(x_test)

print(y_hat[0:10])
print(y_test.values[0:10]) 

"""
[4 4 4 4 4 4 2 2 4 4]
[4 4 4 4 4 4 2 2 4 4]
"""

#%% 
from sklearn import metrics

tree_matrix = metrics.confusion_matrix(y_test, y_hat)
print(tree_matrix)
print('\n')

tree_report = metrics.confusion_matrix(y_test, y_hat) 
print(tree_matrix)
print('\n') 

tree_report = metrics.classification_report(y_test, y_hat)
print(tree_report) 

"""
[[127   4]       # TN(양성을 적확히 예측): 127개, FP(악성으로 잘못 분류한): 4개 
 [  2  72]]      # FN(악성을 양성으로 잘못 분류): 2개, TP(악성을 정확하게 예측): 72개 


[[127   4]
 [  2  72]]


              precision    recall  f1-score   support

           2       0.98      0.97      0.98       131    # 양성 종양 목표값 2
           4       0.95      0.97      0.96        74    # 악성 종양 4 

    accuracy                           0.97       205
   macro avg       0.97      0.97      0.97       205     # 예측값 0.98, 0.96
weighted avg       0.97      0.97      0.97       205     # Av = 0.97 
"""

















#%% 
# 군집: 분석은 데이터셋의 관측값이 갖고 있는 여러 속성을 분석하여 서로 비슷한 특징을 갖는
#      관측값끼리 같은 클러스트로 묶는 알고리즘
# 특이 데이터를 찾는데 활용 
# 비지도학습 
# 신용카드 부정 사용 탐지, 구매 패턴 분석, 소비자 행동 특성 그룹화 use 
#%%
# K-Means 알고리즘 
# 데이터간의 유사성을 측정 
# 백터 공간 --> k개의 클러스터가 주어질때 클러스트의 중심까지 거리가 가장 가까운 클러스터로 해당 데이터를 할당 
# k영향으로 모형에 성능이 평가됨 

import pandas as pd
import matplotlib.pyplot as plt

uci_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/\
00292/Wholesale%20customers%20data.csv'

df = pd.read_csv(uci_path, header=0)

print(df.head())
print('\n')

print(df.head())
print('\n')

print(df.describe()) 

"""
 Channel  Region  Fresh  Milk  Grocery  Frozen  Detergents_Paper  Delicassen
0        2       3  12669  9656     7561     214              2674        1338
1        2       3   7057  9810     9568    1762              3293        1776
2        2       3   6353  8808     7684    2405              3516        7844
3        1       3  13265  1196     4221    6404               507        1788
4        2       3  22615  5410     7198    3915              1777        5185


   Channel  Region  Fresh  Milk  Grocery  Frozen  Detergents_Paper  Delicassen
0        2       3  12669  9656     7561     214              2674        1338
1        2       3   7057  9810     9568    1762              3293        1776
2        2       3   6353  8808     7684    2405              3516        7844
3        1       3  13265  1196     4221    6404               507        1788
4        2       3  22615  5410     7198    3915              1777        5185


          Channel      Region          Fresh          Milk       Grocery  \
count  440.000000  440.000000     440.000000    440.000000    440.000000   
mean     1.322727    2.543182   12000.297727   5796.265909   7951.277273   
std      0.468052    0.774272   12647.328865   7380.377175   9503.162829   
min      1.000000    1.000000       3.000000     55.000000      3.000000   
25%      1.000000    2.000000    3127.750000   1533.000000   2153.000000   
50%      1.000000    3.000000    8504.000000   3627.000000   4755.500000   
75%      2.000000    3.000000   16933.750000   7190.250000  10655.750000   
max      2.000000    3.000000  112151.000000  73498.000000  92780.000000   

             Frozen  Detergents_Paper    Delicassen  
count    440.000000        440.000000    440.000000  
mean    3071.931818       2881.493182   1524.870455  
std     4854.673333       4767.854448   2820.105937  
min       25.000000          3.000000      3.000000  
25%      742.250000        256.750000    408.250000  
50%     1526.000000        816.500000    965.500000  
75%     3554.250000       3922.000000   1820.250000  
max    60869.000000      40827.000000  47943.000000  
"""

#%%

x = df.iloc[:, :]  # 분석 사용할 속성 선택 
print(x[:5])
print('\n') 


# 설명 변수 정규화 
from sklearn import preprocessing
x = preprocessing.StandardScaler().fit(x).transform(x)
 
# standarscaler()함수등을 이용하여 학습데이터 정규화 
# 서로 다른 변수 사이에 존재할 수 있는 데이터 값의 상대적 크기 차이에서 발생하는 오류 제거  
                                                         

print(x[:5]) 

"""
Channel  Region  Fresh  Milk  Grocery  Frozen  Detergents_Paper  Delicassen
0        2       3  12669  9656     7561     214              2674        1338
1        2       3   7057  9810     9568    1762              3293        1776
2        2       3   6353  8808     7684    2405              3516        7844
3        1       3  13265  1196     4221    6404               507        1788
4        2       3  22615  5410     7198    3915              1777        5185

# 0~4까지에 data추출 

[[ 1.44865163  0.59066829  0.05293319  0.52356777 -0.04111489 -0.58936716
  -0.04356873 -0.06633906]
 [ 1.44865163  0.59066829 -0.39130197  0.54445767  0.17031835 -0.27013618
   0.08640684  0.08915105]
 [ 1.44865163  0.59066829 -0.44702926  0.40853771 -0.0281571  -0.13753572
   0.13323164  2.24329255]
 [-0.69029709  0.59066829  0.10011141 -0.62401993 -0.3929769   0.6871443
  -0.49858822  0.09341105]
 [ 1.44865163  0.59066829  0.84023948 -0.05239645 -0.07935618  0.17385884
  -0.23191782  1.29934689]]
"""

#%% 

x = df.iloc[:, :]
print(x[:5]) 
print('\n') 

from sklearn import preprocessing
x = preprocessing.StandardScaler().fit(x).transform(x) 

print(x[:5]) 
"""
Channel  Region  Fresh  Milk  Grocery  Frozen  Detergents_Paper  Delicassen
0        2       3  12669  9656     7561     214              2674        1338
1        2       3   7057  9810     9568    1762              3293        1776
2        2       3   6353  8808     7684    2405              3516        7844
3        1       3  13265  1196     4221    6404               507        1788
4        2       3  22615  5410     7198    3915              1777        5185


[[ 1.44865163  0.59066829  0.05293319  0.52356777 -0.04111489 -0.58936716
  -0.04356873 -0.06633906]
 [ 1.44865163  0.59066829 -0.39130197  0.54445767  0.17031835 -0.27013618
   0.08640684  0.08915105]
 [ 1.44865163  0.59066829 -0.44702926  0.40853771 -0.0281571  -0.13753572
   0.13323164  2.24329255]
 [-0.69029709  0.59066829  0.10011141 -0.62401993 -0.3929769   0.6871443
  -0.49858822  0.09341105]
 [ 1.44865163  0.59066829  0.84023948 -0.05239645 -0.07935618  0.17385884
  -0.23191782  1.29934689]]
"""

#%% 
from sklearn import cluster # cluster 군집 모형 가져오기 

# 모형 객체 생성
kmeans = cluster.KMeans(init='k-means++', n_clusters=5, n_init=10) 
# n_clusters: kMeans 클러스팅 알고리즘을 사용할 때 설정할 수 있는 매개변수 --> 데이터 5개 클러스터로 그룹화하도록 지시 
# n_init=10: == --> 알고리즘 10번 실행하여 각각 다른 초기 중심점에서 start 

# 모형 학습
kmeans.fit(x) 

# 예측(군집)
cluster_label = kmeans.labels_
print(cluster_label)
print('\n')


# 예측결과 데이터프레임 추가 
df['Cluster'] = cluster_label
print(df.head()) 


"""
[1 1 1 2 1 1 1 1 2 1 1 1 1 1 1 2 1 2 1 2 1 2 2 4 1 1 2 2 1 2 2 2 2 2 2 1 2
 1 1 2 2 2 1 1 1 1 1 4 1 1 2 2 1 1 2 2 4 1 2 2 1 4 1 1 2 4 2 1 2 2 2 2 2 1
 1 2 2 1 2 2 2 1 1 2 1 4 4 2 2 2 2 2 4 2 1 2 1 2 2 2 1 1 1 2 2 2 1 1 1 1 2
 1 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2
 2 2 2 2 2 2 2 1 1 2 1 1 1 2 2 1 1 1 1 2 2 2 1 1 2 1 2 1 2 2 2 2 2 4 2 3 2
 2 2 2 1 1 2 2 2 1 2 2 0 1 0 0 1 1 0 0 0 1 0 0 0 1 0 4 0 0 1 0 1 0 1 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 4 0 0 0 0 0 0 0
 0 0 0 0 0 1 0 1 0 1 0 0 0 0 2 2 2 2 2 2 1 2 1 2 2 2 2 2 2 2 2 2 2 2 1 0 1
 0 1 1 0 1 1 1 1 1 1 1 0 0 1 0 0 1 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0
 4 0 1 0 0 0 0 1 1 2 1 2 2 1 1 2 1 2 1 2 1 2 2 2 1 2 2 2 2 2 2 2 1 2 2 2 2
 2 2 2 1 2 2 1 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2
 1 1 2 2 2 2 2 2 1 1 2 1 2 2 1 2 1 1 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2]


   Channel  Region  Fresh  Milk  Grocery  Frozen  Detergents_Paper  \
0        2       3  12669  9656     7561     214              2674   
1        2       3   7057  9810     9568    1762              3293   
2        2       3   6353  8808     7684    2405              3516   
3        1       3  13265  1196     4221    6404               507   
4        2       3  22615  5410     7198    3915              1777   

   Delicassen  Cluster  
0        1338        1  
1        1776        1  
2        7844        1  
3        1788        2  
4        5185        1  

# 관측값 5개의 클러스터 구분 
"""

#%% 
# 2개의 변수 관측값 분포 
df.plot(kind='scatter', x = 'Grocery', y='Frozen', c='Cluster', cmap='Set1',
        colorbar=False, figsize=(10,10)) 

df.plot(kind='scatter', x = 'Milk', y = 'Delicassen', c ='Cluster', cmap='Set1',
        colorbar=True, figsize=(10, 10)) 

plt.show()
plt.close() 















