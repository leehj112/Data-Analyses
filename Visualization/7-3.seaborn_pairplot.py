# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:10:43 2024

@author: leehj
"""

# 라이브러리 불러오기
import matplotlib.pyplot as plt
import seaborn as sns
 
# Seaborn 제공 데이터셋 가져오기
titanic = sns.load_dataset('titanic')
 
# 스타일 테마 설정 (5가지: darkgrid, whitegrid, dark, white, ticks)
sns.set_style('whitegrid')

# titanic 데이터셋 중에서 분석 데이터 선택하기
titanic_pair = titanic[['age','pclass', 'fare']]

# 전달되는 데이터프레임의 열(변수)을 두 개씩 짝을 지을 수 있는 모든 조합에 대해 표현
# 컬럼일 3개 : 3 * 3 -> 9개
# 같은 변수: 히스토그램, 대각선 방향
# 다른 변수: 산점도
# 조건에 따라 그리드 나누기
g = sns.pairplot(titanic_pair)