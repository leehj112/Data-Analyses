# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 23:46:09 2024

@author: dlgur
"""

# -*- coding: utf-8 -*-


# LabelEncoder를 사용하여 각 범주를 고유한 숫자로 변환하는 방법입니다.

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 데이터프레임 생성
data = {'category': ['apple', 'banana', 'orange', 'apple', 'orange', 'pineapple']}
df = pd.DataFrame(data)

# LabelEncoder 객체 생성
label_encoder = LabelEncoder()

# 범주형 데이터를 수치형으로 변환
df['category_encoded'] = label_encoder.fit_transform(df['category'])

print(df)