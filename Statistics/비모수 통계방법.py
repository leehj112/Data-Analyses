# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:34:36 2024

@author: leehj
"""

"""
정규분포가 아닌 데이터를 분석하는 방법 - 비모수 통계 방법

비모수 검정(예: Mann-Whitney U 검정, Kruskal-Wallis 검정, Wilcoxon 부호 순위 검정 등)은 
데이터가 정규분포를 따르지 않을 때 유용
이 방법들은 데이터 분포에 대한 가정을 하지 않기 때문에 정규성 가정이 필요 없습니다.
"""

# Mann-Whitney U 검정
# Mann-Whitney U 검정은 두 독립된 그룹 간의 차이를 비교하는 비모수 검정입니다.

import numpy as np
from scipy.stats import mannwhitneyu

# 예제 데이터
group1 = [14, 15, 16, 19, 20]
group2 = [22, 24, 26, 28, 30]

# Mann-Whitney U 검정 수행
stat, p_value = mannwhitneyu(group1, group2)

print('Mann-Whitney U 검정 통계량:', stat)
print('p-값:', p_value)

#%%

# Kruskal-Wallis 검정
# Kruskal-Wallis 검정은 세 개 이상의 독립된 그룹 간의 차이를 비교하는 비모수 검정

import numpy as np
from scipy.stats import kruskal

# 예제 데이터
group1 = [14, 15, 16, 19, 20]
group2 = [22, 24, 26, 28, 30]
group3 = [10, 12, 14, 16, 18]

# Kruskal-Wallis 검정 수행
stat, p_value = kruskal(group1, group2, group3)

print('Kruskal-Wallis 검정 통계량:', stat)
print('p-값:', p_value)

#%%

# Wilcoxon 부호 순위 검정
# Wilcoxon 부호 순위 검정은 두 관련된 그룹 간의 차이를 비교하는 비모수 검정

import numpy as np
from scipy.stats import wilcoxon

# 예제 데이터
before_treatment = [20, 22, 24, 26, 28]
after_treatment = [21, 23, 25, 27, 29]

# Wilcoxon 부호 순위 검정 수행
stat, p_value = wilcoxon(before_treatment, after_treatment)

print('Wilcoxon 부호 순위 검정 통계량:', stat)
print('p-값:', p_value)

#%%

# 이 예제들은 비모수 검정을 파이썬으로 수행하는 기본적인 방법을 보여준다.
# 데이터에 따라 적절한 검정을 선택하고, 검정 결과를 해석하여 통계적 유의성을 평가할 수 있다.