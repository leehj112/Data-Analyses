# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:08:52 2024

@author: leehj
"""

# -*- coding: utf-8 -*-

# 이웃하고 있는 값으로 바꾸기
# df['embark_town'].fillna(method='ffill', inplace=True)
# method : 
#  - 'ffill' : NaN이 있는 행의 직전 값으로 바꿈
#  - 'bfill' : NaN이 있는 행의 다음 행의 값으로 바꿈


# 라이브러리 불러오기
import seaborn as sns

# titanic 데이터셋 가져오기
df = sns.load_dataset('titanic')
print(df[:][10:20])


#%%

# embark_town 열의 829행의 NaN 데이터 출력
print(df['embark_town'][825:835]) # 829 NaN
print('\n')

#%%
# FutureWarning: Series.fillna with 'method' is deprecated 
# and will raise in a future version.
# embark_town 열의 NaN값을 바로 앞에 있는 828행의 값으로 변경하기
df['embark_town'].fillna(method='bfill', inplace=True)
print(df['embark_town'][825:835])

#%%

# 권고
df['embark_town'].bfill(inplace=True)
print(df['embark_town'][825:835])

#%%

ndf = df['embark_town'].bfill()
print(df['embark_town'][825:835])
print(ndf[825:835])