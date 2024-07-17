# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 23:47:36 2024

@author: dlgur
"""

# Ordinal Encoding
# OrdinalEncoder를 사용하여 각 범주를 순서대로 숫자로 변환하는 방법입니다.

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# 데이터프레임 생성
data = {'category': ['apple', 'banana', 'orange', 'apple', 'orange']}
df = pd.DataFrame(data)

# OrdinalEncoder 객체 생성
ordinal_encoder = OrdinalEncoder()

# 범주형 데이터를 수치형으로 변환
df['category_encoded'] = ordinal_encoder.fit_transform(df[['category']])

print(df)