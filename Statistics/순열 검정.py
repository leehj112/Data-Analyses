# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:51:15 2024

@author: leehj
"""

"""
순열 검정(Permutation Test)은 두 그룹 간의 차이를 검정할 때 사용하는 비모수 방법 중 하나이다.
이 방법은 데이터의 분포에 대한 가정 없이 두 그룹 간의 차이를 비교할 수 있다. 
"""

#%%

import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# 예제 데이터 생성
np.random.seed(0)
group1 = np.random.normal(loc=5, scale=1, size=30)
group2 = np.random.normal(loc=6, scale=1, size=30)

# 두 그룹의 평균 차이 계산
observed_diff = np.mean(group2) - np.mean(group1)
print('관측된 평균 차이:', observed_diff)

#%%

# 순열 검정 함수 정의

def permutation_test(data1, data2, num_permutations=10000):
    # 두 그룹의 결합된 데이터
    combined = np.concatenate([data1, data2])
    
    # 관측된 평균 차이
    observed_diff = np.mean(data2) - np.mean(data1)
    
    count = 0
    for _ in range(num_permutations):
        # 데이터 섞기
        np.random.shuffle(combined)
        
        # 섞인 데이터를 두 그룹으로 나누기
        perm_data1 = combined[:len(data1)]
        perm_data2 = combined[len(data1):]
        
        # 순열 평균 차이
        perm_diff = np.mean(perm_data2) - np.mean(perm_data1)
        
        # 순열 평균 차이가 관측된 평균 차이보다 크거나 같은 경우 카운트 증가
        if perm_diff >= observed_diff:
            count += 1

    # p-값 계산
    p_value = count / num_permutations
    return p_value

#%%

# 순열 검정 수행
p_value = permutation_test(group1, group2)
print('순열 검정 p-값:', p_value)

# 위의 코드는 두 그룹의 데이터를 생성한 후, 순열 검정을 수행하여 p-값을 계산하는 과정을 보여준다. 
# permutation_test 함수는 두 그룹의 데이터를 결합하여 순열 검정을 수행하고, 
# 관측된 평균 차이보다 큰 경우의 수를 카운트하여 p-값을 계산한다.

#%%

# 그래프 한글 깨짐 문제 해결을 위해 맑은고딕 폰트 지정
from matplotlib import font_manager, rc
font_path = "c:/Windows/Fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

#%%


# 순열 검정 시각화
def plot_permutation_test(data1, data2, num_permutations=10000):
    combined = np.concatenate([data1, data2])
    observed_diff = np.mean(data2) - np.mean(data1)
    perm_diffs = []

    for _ in range(num_permutations):
        np.random.shuffle(combined)
        perm_data1 = combined[:len(data1)]
        perm_data2 = combined[len(data1):]
        perm_diff = np.mean(perm_data2) - np.mean(perm_data1)
        perm_diffs.append(perm_diff)

    perm_diffs = np.array(perm_diffs)
    
    plt.hist(perm_diffs, bins=30, alpha=0.75, color='blue', edgecolor='black')
    plt.axvline(observed_diff, color='red', linestyle='dashed', linewidth=2)
    plt.title('순열 검정 결과')
    plt.xlabel('평균 차이')
    plt.ylabel('빈도수')
    plt.show()

#%%

# 이 시각화 코드는 순열 검정의 결과를 히스토그램으로 보여주며, 
# 관측된 평균 차이를 빨간색 선으로 표시한다. 
# 이를 통해 순열 검정의 결과를 시각적으로 확인할 수 있다.
plot_permutation_test(group1, group2)