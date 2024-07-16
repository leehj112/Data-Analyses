# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 12:16:14 2024

@author: leehj
"""

# 비지도학습(Unsupervised Learning)
# 군집(clustering) : k-Means, DBSCAN 
# DBSCAN(Density Based Spatial Custering of Applications with Noise)
# 공간의 밀집도
# 코어포인트(Core Point): 자기를 중심으로 반지름 R의 공간에 최소 M개의 포인트가 존재하는 점
# 경계포인트(Border Point): 반지름 R안에 다른 코어 포인트가 있을 경우
# 로이즈(Noise): outlier, 코어 포인트도 아니고 경계 포인트에도 속하지 않는 점

### 기본 라이브러리 불러오기
import pandas as pd

import folium


#%%
'''
[Step 1] 데이터 준비/ 기본 설정
'''

# 서울시내 중학교 진학률 데이터셋 (출처: 교육???)
file_path = './2016_middle_shcool_graduates_report.xlsx'
df = pd.read_excel(file_path, engine='openpyxl', header=0)

# IPython Console 디스플레이 옵션 설정하기
pd.set_option('display.width', None)        # 출력화면의 너비
pd.set_option('display.max_rows', 100)      # 출력할 행의 개수 한도
pd.set_option('display.max_columns', 10)    # 출력할 열의 개수 한도
pd.set_option('display.max_colwidth', 20)   # 출력할 열의 너비
pd.set_option('display.unicode.east_asian_width', True)   # 유니코드 사용 너비 조정

#%%

df.drop('Unnamed: 0', axis=1, inplace=True)

#%%
# 열 이름 배열을 출력
print(df.columns.values)   
print('\n')

#%%
'''
[Step 2] 데이터 탐색
'''

# 데이터 살펴보기
print(df.head())   
print('\n')

#%%
# 데이터 자료형 확인
print(df.info())  
print('\n')

#%%
# 데이터 통계 요약정보 확인
print(df.describe())
print('\n')

#%%

help(folium.Map)

#%%

# 지도에 위치 표시
# mschool_map = folium.Map(location=[37.55,126.98], tiles='Stamen Terrain', zoom_start=12)
mschool_map = folium.Map(location=[37.55,126.98], zoom_start=12)


# 중학교 위치정보를 CircleMarker로 표시
for name, lat, lng in zip(df.학교명, df.위도, df.경도):
    folium.CircleMarker([lat, lng],
                        radius=10,             # 원의 반지름
                        color='brown',         # 원의 둘레 색상
                        fill=True,
                        fill_color='coral',    # 원을 채우는 색
                        fill_opacity=0.7,      # 투명도    
                        popup=name
    ).add_to(mschool_map)

# 지도를 html 파일로 저장하기
mschool_map.save('./seoul_mschool_location.html')

#%%

'''
[Step 3] 데이터 전처리
'''

# 원핫인코딩(더미 변수)
from sklearn import preprocessing    

label_encoder = preprocessing.LabelEncoder()     # label encoder 생성
onehot_encoder = preprocessing.OneHotEncoder()   # one hot encoder 생성

onehot_location = label_encoder.fit_transform(df['지역'])
onehot_code = label_encoder.fit_transform(df['코드'])
onehot_type = label_encoder.fit_transform(df['유형'])
onehot_day = label_encoder.fit_transform(df['주야'])

df['location'] = onehot_location
df['code'] = onehot_code
df['type'] = onehot_type
df['day'] = onehot_day

print(df.head())   
print('\n')


#%%
'''
[Step 4] DBSCAN 군집 모형 - sklearn 사용
'''

# sklearn 라이브러리에서 cluster 군집 모형 가져오기 
from sklearn import cluster

# 분석에 사용할 속성을 선택 (과학고, 외고국제고, 자사고 진학률)
columns_list = [9, 10, 13]
X = df.iloc[:, columns_list]
print(X[:5])
print('\n')

# 설명 변수 데이터를 정규화
X = preprocessing.StandardScaler().fit(X).transform(X)

#%%
# DBSCAN 모형 객체 생성
# eps : 밀도계산의 기준이 되는 반지름(R)
# min_samples : 최소 포인트 갯수(M)
dbm = cluster.DBSCAN(eps=0.2, min_samples=5)

# 모형 학습
dbm.fit(X)   
 
# 예측 (군집) 
cluster_label = dbm.labels_   
print(cluster_label)
print('\n')

# 예측 결과를 데이터프레임에 추가
df['Cluster'] = cluster_label
print(df.head())   
print('\n')

#%%

cluster_value_counts = df['Cluster'].value_counts().sort_index()
print(cluster_value_counts)

#%%
"""
Cluster
-1    255
 0    102
 1     45
 2      8
 3      5
Name: count, dtype: int64
"""

#%%
# 클러스터 값으로 그룹화하고, 그룹별로 내용 출력 (첫 5행만 출력)
# 진학률: 과학고, 외고국제고, 자사고 진학률
grouped_cols = [0, 1, 3] + columns_list # [9,10,13] 
print(grouped_cols) # [0, 1, 3, 9, 10, 13]

#%%
# [0, 1, 3, 9, 10, 13]
# ['지역', '학교명', '유형', '과학고', '외고국제고', '자사고']

#%%
grouped = df.groupby('Cluster')
for key, group in grouped:
    print('* key :', key)
    print('* number :', len(group))    
    print(group.iloc[:, grouped_cols].head())
    print('\n')

#%%
# 그래프로 표현 - 시각화
colors = {-1:'gray', 0:'coral', 1:'blue', 2:'green', 3:'red', 4:'purple', 
          5:'orange', 6:'brown', 7:'brick', 8:'yellow', 9:'magenta', 10:'cyan', 11:'tan'}

# cluster_map = folium.Map(location=[37.55,126.98], tiles='Stamen Terrain', zoom_start=12)
cluster_map = folium.Map(location=[37.55,126.98], zoom_start=12)

for name, lat, lng, clus in zip(df.학교명, df.위도, df.경도, df.Cluster):  
    folium.CircleMarker([lat, lng],
                        radius=5,                   # 원의 반지름
                        color=colors[clus],         # 원의 둘레 색상
                        fill=True,
                        fill_color=colors[clus],    # 원을 채우는 색
                        fill_opacity=0.7,           # 투명도    
                        popup=name
    ).add_to(cluster_map)

# 지도를 html 파일로 저장하기
cluster_map.save('./seoul_mschool_cluster.html')

#%%


# X2 데이터셋에 대하여 위의 과정을 반복(과학고, 외고국제고, 자사고 진학률 + 유형)
columns_list2 = [9, 10, 13, 22]
X2 = df.iloc[:, columns_list2]
print(X2[:5])
print('\n')


X2 = preprocessing.StandardScaler().fit(X2).transform(X2)
dbm2 = cluster.DBSCAN(eps=0.2, min_samples=5)
dbm2.fit(X2)  
df['Cluster2'] = dbm2.labels_   

grouped2_cols = [0, 1, 3] + columns_list2
grouped2 = df.groupby('Cluster2')
for key, group in grouped2:
    print('* key :', key)
    print('* number :', len(group))    
    print(group.iloc[:, grouped2_cols].head())
    print('\n')

# cluster2_map = folium.Map(location=[37.55,126.98], tiles='Stamen Terrain', zoom_start=12)
cluster2_map = folium.Map(location=[37.55,126.98], zoom_start=12)

for name, lat, lng, clus in zip(df.학교명, df.위도, df.경도, df.Cluster2):  
    folium.CircleMarker([lat, lng],
                        radius=5,                   # 원의 반지름
                        color=colors[clus],         # 원의 둘레 색상
                        fill=True,
                        fill_color=colors[clus],    # 원을 채우는 색
                        fill_opacity=0.7,           # 투명도    
                        popup=name
    ).add_to(cluster2_map)

# 지도를 html 파일로 저장하기
cluster2_map.save('./seoul_mschool_cluster2.html')


# X3 데이터셋에 대하여 위의 과정을 반복(과학고, 외고_국제고)
columns_list3 = [9, 10]
X3 = df.iloc[:, columns_list3]
print(X3[:5])
print('\n')

X3 = preprocessing.StandardScaler().fit(X3).transform(X3)
dbm3 = cluster.DBSCAN(eps=0.2, min_samples=5)
dbm3.fit(X3)  
df['Cluster3'] = dbm3.labels_   

grouped3_cols = [0, 1, 3] + columns_list3
grouped3 = df.groupby('Cluster3')
for key, group in grouped3:
    print('* key :', key)
    print('* number :', len(group))    
    print(group.iloc[:, grouped3_cols].head())
    print('\n')

# cluster3_map = folium.Map(location=[37.55,126.98], tiles='Stamen Terrain', zoom_start=12)
cluster3_map = folium.Map(location=[37.55,126.98], zoom_start=12)

for name, lat, lng, clus in zip(df.학교명, df.위도, df.경도, df.Cluster3):  
    folium.CircleMarker([lat, lng],
                        radius=5,                   # 원의 반지름
                        color=colors[clus],         # 원의 둘레 색상
                        fill=True,
                        fill_color=colors[clus],    # 원을 채우는 색
                        fill_opacity=0.7,           # 투명도    
                        popup=name
    ).add_to(cluster3_map)

# 지도를 html 파일로 저장하기
cluster3_map.save('./seoul_mschool_cluster3.html')

#%%
# THE END