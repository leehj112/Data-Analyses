# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:34:39 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

###############################################################################
# 데이터 프레임 집계 및 병합
###############################################################################

#%%

# 데이터 프레임 집계

#%%

# iris 데이터셋(데이터프레임) 불러오기
import pandas as pd
import seaborn as sns
df = sns.load_dataset('iris')
df.head()


#%%

# species별 평균
df.groupby('species').mean()

#%%

# species별 최댓값
df.groupby('species').max()


#%%

# 주요 통계량을 한 번에 출력하는 함수 describe()
df.groupby('species').describe()

#%%

# species별 통계량 표시(행열변환)
df.groupby('species').describe().T


#%%

###############################################################################
# 데이터 프레임 병합
###############################################################################

#%%

# 실습용 데이터셋1 만들기
import pandas as pd
df1 = pd.DataFrame({
    'city' : ['서울','부산','대구','대전','광주'],
    'pop' : [978, 343, 247, 153, 150]
})

#%%

# 실습용 데이터셋2 만들기
df2 = pd.DataFrame({
    'city' : ['서울','부산','대구','인천'],
    'area' : [605, 770, 884, 1063]
})


#%%

df1

#%%

"""
  city  pop
0   서울  978
1   부산  343
2   대구  247
3   대전  153
4   광주  150
"""

#%%

df2

#%%
"""
  city  area
0   서울   605
1   부산   770
2   대구   884
3   인천  1063

"""

#%%


#%%

# ① LEFT JOIN

pd.merge(df1, df2, on = 'city', how = 'left')

#%%

"""
  city  pop   area
0   서울  978  605.0
1   부산  343  770.0
2   대구  247  884.0
3   대전  153    NaN
4   광주  150    NaN
"""

#%%

# ② RIGHT JOIN

pd.merge(df1, df2, on = 'city', how = 'right')


#%%

"""
  city    pop  area
0   서울  978.0   605
1   부산  343.0   770
2   대구  247.0   884
3   인천    NaN  1063
"""

#%%

# ③ INNER JOIN

pd.merge(df1, df2, on = 'city')

#%%
"""
  city  pop  area
0   서울  978   605
1   부산  343   770
2   대구  247   884
"""

#%%

# ④ OUTER JOIN

pd.merge(df1, df2, on = 'city', how = 'outer')

#%%

"""
  city    pop    area
0   광주  150.0     NaN
1   대구  247.0   884.0
2   대전  153.0     NaN
3   부산  343.0   770.0
4   서울  978.0   605.0
5   인천    NaN  1063.0
"""

#%%

# THE END

