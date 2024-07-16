# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:13:43 2024

@author: leehj
"""

# - pandas 라이브러리 불러오기

# In[21]:


import pandas as pd
print(pd.__version__)


# - 문자열을 원소로 갖는 1차원 배열 - 리스트


# In[22]:


data1 = ['a', 'b', 'c', 'd', 'e']
print(data1)
print("자료형: ", type(data1))


# - 리스트를 판다스 시리즈로 변환

# In[23]:


sr1 = pd.Series(data1)
print("자료형: ", type(sr1))


# In[24]:


print(sr1)


# In[25]:


sr1.loc[0]


# In[26]:

# 인덱스로 지정하면 시작 인덱스부터 마지막 인덱스까지 선택
sr1.loc[1:3]
"""
1    b
2    c
3    d
dtype: object
"""

# - 투플을 판다스 시리즈로 변환

# In[27]:


data2 = (1, 2, 3.14, 100, -10)
sr2 = pd.Series(data2)
print(sr2)


# - 시리즈를 결합하여 데이터프레임으로 변환

# In[28]:


dict_data = {'c0':sr1, 'c1':sr2} 
df1 = pd.DataFrame(dict_data)  
df1


# In[29]:


type(df1)


# - 데이터프레임의 열(column)

# In[30]:


df1.columns


# In[31]:


df1.columns = ['string', 'number']
df1


# - 데이터프레임의 행 인덱스(row index)

# In[32]:


df1.index


# In[33]:


df1.index = ['r0', 'r1', 'r2', 'r3', 'r4']
df1


# - 데이터프레임의 원소 선택

# In[34]:


# 인덱스, 칼럼으로 선택
df1.loc['r2', 'number']


# - 데이터프레임의 부분 추출

# In[35]:

# 행의 범위, 열의 범위 지정해서 선택
df1.loc['r2':'r3', 'string':'number']


# In[36]:

# 행의 범위 열을 1개 지정해서 선택
df1.loc['r2':'r3', 'number']


# In[37]:

# 행은 1개 지정, 칼럼은 범위를 지정해서 선택
df1.loc['r2', 'string':'number']


# In[38]:


# 전체 행의 인덱스를 지정, 열은 1개 지정해서 선택
df1.loc[:, 'string']


# In[39]:

# 행의 인덱스의 범위를 지정, 열은 전체를 지정해서 선택
df1.loc['r2':'r3', :]