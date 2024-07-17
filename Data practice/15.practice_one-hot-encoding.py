# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 23:46:56 2024

@author: dlgur
"""

# -*- coding: utf-8 -*-

# One-Hot Encoding
# get_dummies를 사용하여 각 범주를 별도의 열로 변환하는 방법입니다.

import pandas as pd

# 데이터프레임 생성
data = {'category': ['apple', 'banana', 'orange', 'apple', 'orange']}
df = pd.DataFrame(data)

# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['category'])

print(df_encoded)