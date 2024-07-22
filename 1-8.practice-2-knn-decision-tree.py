# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:28:37 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# # [06] 분류 및 군집분석

###############################################################################
# ## 2. 병아리의 품종을 구분할 수 있을까? (분류 알고리즘)
###############################################################################

# [분석 스토리]  
# 무럭무럭 잘 자라고 있는 병아리들을 관찰하던 어느 날, 뭔가 특이한 점을 발견했습니다.
# 병아리가 성장함에 따라 생김새가 변하겠지만 병아리들마다 
# 날개 길이, 꽁지깃 길이, 볏의 높이가 유난히 차이가 나 보였습니다. 
# 불안한 마음에 사진을 찍어 종란 판매업체 담당자에게 이 병아리들이 같은 품종이 맞는지 문의했습니다. 
# 그리고 담당자에게서 온 답변은 매우 혼란스럽게 만들었습니다. 
# 종란 판매 직원의 실수로 주문을 넣었던 A 품종의 종란뿐만 
# 아니라 B와 C라는 2가지 품종의 종란이 섞여서 납품되었다는 것입니다. 
# 졸지에 3가지 품종의 병아리를 키우게 되어 판매처에 클레임(claim)을 제기했고, 
# 종란 판매처의 품종 엔지니어가 양계농장에 급파되었습니다. 
# 엔지니어는 하루에 걸쳐 총 300마리의 병아리 품종을 정확히 구분해 기록했습니다. 
# 다음에도 혹시나 이런 일이 발생할지 모른다는 불안감에 
# 이 300마리의 병아리 데이터를 활용해 품종을 구분할 수 있는 분류 모델을 개발해 보려고 합니다. 
# 과연 그는 암수를 구분했던 것처럼 품종도 잘 구분해낼 수 있을까요?

#%%
# ### 2-1. 다양한 분류 알고리즘

# ### 2-2. 나이브 베이즈 분류

# 나이브 베이즈 분류(Naïve Bayes Classification)는 베이즈 정리를 적용한 확률 분류 기법입니다. 
# 베이즈 정리는 쉽게 말해 조건부 확률을 구하는 공식으로 생각하면 됩니다. 
# 조건부 확률은 사건 B가 일어났다는 조건하에 사건 A가 일어날 확률을 P(A|B)라고 표현하는데 사후확률(posterior)이라고도 합니다. 
# 식으로 나타내면 다음과 같습니다.
#   
# > $$ P(A|B) = {P(A \cap B) \over P(B)} = {P(B|A)P(A) \over P(B)} = posterior = {likelihood×prior \over evidence} $$
#   
# > 여기서 P(A), P(B)는 각각 사건 A, B가 일어날 확률이고, 
# P(B|A)는 사건 A가 일어난다는 조건하에 사건 B가 일어날 확률을 나타내며 우도(likelihood)라고 부릅니다. 
# 베이즈 정리는 사건 B가 발생(P(B)=1)함으로써 사건 A가 발생할 확률이 
# 어떻게 변하는지를 표현한 식으로 B라는 사건을 관찰해 A라는 사건에 
# 어떤 영향을 미치는지 찾아내는 방법이라고 이해하면 될 것 같습니다.

# In[21]:

# 실습용 데이터
# 지도학습 사례(Supervised Learning)    
# 병아리 300마리 품종 데이터를 훈련 데이터 테스트 용도의 2가지 데이터 셋으로 8:2 비율로 분할

# 실습용 데이터 불러오기
# wing_length : 날개 길이
# tail_length : 꽁지깃 길이
# comb_height : 볏 높이
# breeds      : 품종('a', 'b', 'c')

import pandas as pd
df_train = pd.read_csv('dataset/ch6-2_train.csv')
df_test = pd.read_csv('dataset/ch6-2_test.csv')


# In[22]:

# 훈련 데이터: 240건
df_train.info()


# In[23]:

# 테스트 데이터: 60건
df_test.info()


# In[24]:


df_train.head()

#%%

df_train['breeds'].value_counts() # a:80, b:80, c:80

#%%

df_test['breeds'].value_counts() # a:20, b:20, c:20


# In[25]:


# train, test 데이터셋 각각 x, y로 분할, ndarray 타입
x_train = df_train.iloc[:,0:3].values
y_train = df_train.iloc[:,3].values
x_test = df_test.iloc[:,0:3].values
y_test = df_test.iloc[:,3].values


# In[26]:


# 나이브베이즈 알고리즘 수행을 위한 함수 불러오기
from sklearn.naive_bayes import GaussianNB


# In[27]:


# 모델 구축 및 학습
model_nb = GaussianNB().fit(x_train, y_train)


# In[28]:


# 예측값 생성
y_pred_nb = model_nb.predict(x_test)


# In[29]:


# 예측값 확인
y_pred_nb


# In[30]:


# 예측 결과 평가
from sklearn.metrics import confusion_matrix
# 위쪽이 예측값, 좌측이 실제값
confusion_matrix(y_test, y_pred_nb)


# In[31]:


# 예측 결과 평가
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_nb))

#%%

# 테스트 : 60건
# 정답 : 56건 = a(20) + b(17) + c(19)
# 정확도(Accuracy) : 93.33%

"""
              precision    recall  f1-score   support

           a       0.95      1.00      0.98        20
           b       0.94      0.85      0.89        20
           c       0.90      0.95      0.93        20

    accuracy                           0.93        60
   macro avg       0.93      0.93      0.93        60
weighted avg       0.93      0.93      0.93        60
"""


#%%

###############################################################################
# ### 2-3. k-NN
###############################################################################

# k-최근접 이웃(k-NN, k-Nearest Neighbor)은 머신러닝 알고리즘으로 
# 새로운 데이터에 대해 이와 가장 거리가 가까운 k개의 과거 데이터의 결과를 이용해
# 다수결로 분류하는 방법이다.

# <div>
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/KnnClassification.svg/558px-KnnClassification.svg.png" width="500"/>
#     <center>k-NN 알고리즘(출처 : wikipedia)</center>
# </div>

# > 원(●)은 새로운 데이터인데, 과거 데이터를 이용해 네모(■) 또는 세모(▲)로 분류
# 실선으로 된 원은 k가 3개인 경우입니다. 
# 이때에는 실선 원 안에 네모(■) 1개, 세모(▲) 2개가 있습니다. 
# 이 경우 원(●)은 다수결에 의해 개수가 더 많은 세모(▲)로 분류됩니다. 
# 점선으로 된 원은 k가 5개인 경우입니다. 
# 이때에는 점선 안에 네모(■) 3개, 세모(▲) 2개가 있습니다. 
# 이 경우 원(●)은 다수결에 의해 개수가 더 많은 네모(■)로 분류됩니다.  
# 이렇게 k값의 선택에 따라 새로운 데이터에 대한 분류 결과가 달라지며, 
# 종속변수의 형태(범주형 또는 연속형)에 따라 분류(classification)와 회귀(regression) 모두에 사용할 수 있습니다. 
# 그리고 새 데이터에 더 가까운 이웃일수록 더 먼 이웃보다 평균에 더 많이 기여하도록 
# 이웃의 기여에 가중치(weight)를 줄 수 있습니다. 
# 예를 들어, 이웃까지의 거리가 d라면 해당 이웃들에게는 거리의 반비례인 1/d만큼의 가중치를 부여할 수 있습니다.

# In[32]:


# k-NN 알고리즘 수행을 위한 함수 불러오기
from sklearn.neighbors import KNeighborsClassifier


# In[33]:

# 훈련 데이터를 스케일링을 하지 않고 학습을 함
# 그러나 k-NN은 거리 기반 알고리즘이기 때문에 독립변수를 스케일링 한 후 학습하는 것이 좋음

# 모델 구축(k=3) 및 학습
model_knn = KNeighborsClassifier(n_neighbors = 3).fit(x_train,y_train)


# In[34]:


y_pred_knn = model_knn.predict(x_test)


# In[35]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred_knn)

#%%

# 위쪽이 예측값, 좌측이 실제값

"""
array([[19,  1,  0],
       [ 1, 16,  3],
       [ 0,  1, 19]], dtype=int64)
"""


# In[36]:

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_knn))

#%%

# 이웃의 갯수: k=3
# 정확도 : 90%
"""
              precision    recall  f1-score   support

           a       0.95      0.95      0.95        20
           b       0.89      0.80      0.84        20
           c       0.86      0.95      0.90        20

    accuracy                           0.90        60
   macro avg       0.90      0.90      0.90        60
weighted avg       0.90      0.90      0.90        60


"""

# In[37]:


# 모델 구축(k=5) 및 학습
model_knn5 = KNeighborsClassifier(n_neighbors = 5).fit(x_train,y_train)


# In[38]:


y_pred_knn5 = model_knn5.predict(x_test)


# In[39]:


# 위쪽이 예측값, 좌측이 실제값
confusion_matrix(y_test, y_pred_knn5)

#%%

"""
array([[20,  0,  0],
       [ 1, 17,  2],
       [ 0,  2, 18]], dtype=int64)
"""


# In[40]:


print(classification_report(y_test, y_pred_knn5))

#%%

# 이웃의 갯수: k=5
# 정확도 : 92%

"""
              precision    recall  f1-score   support

           a       0.95      1.00      0.98        20
           b       0.89      0.85      0.87        20
           c       0.90      0.90      0.90        20

    accuracy                           0.92        60
   macro avg       0.92      0.92      0.92        60
weighted avg       0.92      0.92      0.92        60
"""

#%%

###############################################################################
# ### 2-4. 의사결정나무
###############################################################################

# 의사결정나무(Decision Tree)는 주어진 독립변수에 의사결정규칙을 적용해 나가면서 종속변수를 예측해 나가는 알고리즘입니다. 
# 종속변수가 범주형이나 연속형인 경우 모두 사용할 수 있고, 분석 결과가 조건 형태로 출력되기 때문에 모델을 이해하기가 쉽습니다.  

# 다음은 타이타닉호에서 생존 여부(종속변수)를 성별, 나이, 함께 탑승한 형제 또는 배우자수(sibsp)와 같은 
# 다양한 독립변수에 의사결정규칙을 적용해 트리 형태로 나타낸 의사결정나무 모델입니다.  

# <div>
# <img src="https://upload.wikimedia.org/wikipedia/commons/f/fe/CART_tree_titanic_survivors_KOR.png" width="500"/>
#     <center>의사결정나무 알고리즘(출처 : wikipedia)</center>
# </div>

# 위 그림에서 마름모 형태로 표현되는 노드를 의사결정 노드(Decision Node)라고 하고, 
# 타원 형태로 표현되는 노드를 잎사귀 노드(Leaf Node)라고 합니다. 
# 의사결정 노드 중 최초로 분류가 시작되는 최상단의 노드를 뿌리 노드(Root Node)라고 합니다.  
# 의사결정나무는 종속변수가 범주형인 경우 분류나무(Classification Tree), 
# 연속형인 경우 회귀나무(Regression Tree)로 구분된다.
# 
# 의사결정규칙(가지를 분할할 때)을 만들 때 기준이 될 독립변수 항목과 값을 선택하는 방법으로 분류나무는 χ2 통계량의 p값, 
# 지니 지수(Gini Index), 엔트로피 지수(Entropy Index) 등을 사용하고, 회귀나무는 F 통계량의 p값, 분산의 감소량 등을 사용합니다.  
# 의사결정나무 알고리즘에는 CART, CHAID, ID3, C4.5, C5.0, MARS 등의 다양한 방법론이 존재합니다. 

# sklearn 패키지는 최적화된 버전의 CART(Classification And Regression Tree) 알고리즘을 이용합니다.

# 불손도와 정보이득
# 분류 트리의 분할 기준(criterion)은 지니(Gini)와 엔트로피(Entropy) 지수는 불순도(impurity)를 측정하는 방법이다.
# 의사결정나무 알고리즘은 부모와 자식 노드의 불순도 차이가 크도록 트리를 성장시키는데 
# 불손도 차이를 정보 이득(Information Gain)이라고 부른다.
#
# 불순도와 정보이득을 계산하는 방법
#   - 지니 불순도  $$1-\sum_{i=1}^n(p_i)^2 \; (p_i : i\;클래스로 \;분류되는 확률)$$
#   - 엔트로피 불순도  $$-\sum_{i=1}^n(p_i)log_2(p_i) \; (p_i : i\;클래스로 \;분류되는 확률)$$
#   - 정보이득  $$부모노드 \; 불순도 - \sum_{i=1}^n(m_i) \; (m_i : i\;클래스 \; 자식 노드의 \;가중 평균 \;불순도)$$


# In[41]:


# 의사결정나무 알고리즘 수행 및 시각화를 위한 함수 불러오기
from sklearn.tree import DecisionTreeClassifier, plot_tree


# In[42]:


# 모델 구축 및 학습
# 기본 분할 기준(criterion) : 지니 지수
model_tree = DecisionTreeClassifier().fit(x_train, y_train)


# In[43]:


y_pred_tree = model_tree.predict(x_test)


# In[44]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred_tree)

#%%

"""
array([[20,  0,  0],
       [ 1, 18,  1],
       [ 0,  4, 16]], dtype=int64)
"""


# In[45]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_tree))

#%%

# 정확도: 0.90
# 민감도: 0.80(최소)

"""
              precision    recall  f1-score   support

           a       0.95      1.00      0.98        20
           b       0.82      0.90      0.86        20
           c       0.94      0.80      0.86        20

    accuracy                           0.90        60
   macro avg       0.90      0.90      0.90        60
weighted avg       0.90      0.90      0.90        60
"""

# In[46]:


# 모델 구축(분할기준:entropy) 및 학습
model_tree2 = DecisionTreeClassifier(criterion='entropy').fit(x_train, y_train)


# In[47]:


y_pred_tree2 = model_tree2.predict(x_test)


# In[48]:


confusion_matrix(y_test, y_pred_tree2)

#%%

"""
array([[20,  0,  0],
       [ 1, 18,  1],
       [ 0,  2, 18]], dtype=int64)
"""

# In[49]:


print(classification_report(y_test, y_pred_tree2))

#%%

# 분할 기준이 지니 지수보다는 엔트로피 지수가 더 적합한 것으로 평가

# 정확도 : 0.93
# 민감도 : 0.90(최소값)
# 정밀도 : 0.90(최소값)

"""
              precision    recall  f1-score   support

           a       0.95      1.00      0.98        20
           b       0.90      0.90      0.90        20
           c       0.95      0.90      0.92        20

    accuracy                           0.93        60
   macro avg       0.93      0.93      0.93        60
weighted avg       0.93      0.93      0.93        60
"""

# In[50]:

# 뿌리 노드 : 
#  - 꽁지깃 길이 : tail_length <= 67.5
#  - 훈련용 데이터 셋 : sample = 240
#  - 값(a,b,c) : value[80,80,80], 
# 조건을 만족하면(yes) 좌측, 만족하지 않으면 오른쪽으로 이동
# 

# 트리 그리기
import matplotlib.pyplot as plt
plt.figure(figsize=(15,15))
plot_tree(model_tree2, feature_names = ['wing_length','tail_length','comb_height'], filled = True)
plt.show()


# In[51]:


# 모델 구축(분할기준:entropy, 최대깊이:3) 및 학습
# 트리의 크기(깊이)를 제한해 과적합을 방지하는 방법을 가지치기(pruning)라고 한다.
model_tree3 = DecisionTreeClassifier(criterion='entropy', max_depth = 3).fit(x_train, y_train)


# In[52]:


y_pred_tree3 = model_tree3.predict(x_test)


# In[53]:


confusion_matrix(y_test, y_pred_tree3)

#%%

"""
array([[20,  0,  0],
       [ 1, 14,  5],
       [ 0,  2, 18]], dtype=int64)
"""

# In[54]:


print(classification_report(y_test, y_pred_tree3))

#%%

"""
              precision    recall  f1-score   support

           a       0.95      1.00      0.98        20
           b       0.88      0.70      0.78        20
           c       0.78      0.90      0.84        20

    accuracy                           0.87        60
   macro avg       0.87      0.87      0.86        60
weighted avg       0.87      0.87      0.86        60

"""


# In[55]:


# 트리 그리기
plt.figure(figsize=(15,15))
plot_tree(model_tree3, feature_names = ['wing_length','tail_length','comb_height'], filled = True)
plt.show()

#%%

# THE END