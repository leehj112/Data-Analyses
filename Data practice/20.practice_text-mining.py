# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 23:49:31 2024

@author: dlgur
"""

#!/usr/bin/env python
# coding: utf-8

# [08] 텍스트 마이닝

# [분석스토리]  
# 인터넷 상점을 통해 처음으로 키운 닭 300마리를 판매했다.
# 경쟁사 대비 품질 경쟁력이 떨어진다고 생각해 가격을 낮췄더니 1주일 만에 300마리가 모두 판매되었다. 
# 놀라운 일이었고, 당장 부자가 될 것만 같은 생각에 기쁨을 감출 수가 없었다. 
# 그리고 고객들의 상품 리뷰가 하나씩 달리기 시작했다. 
# 리뷰를 읽고 기쁨과 좌절을 동시에 느끼며 고객의 마음을 읽어 부족한 점을 개선해야겠다고 생각했다. 
# 과연 고객들은 닭을 어떻게 생각하고 있을까?

# 고객 리뷰에서 어떻게 핵심을 파악할 수 있을까?

# 워드 클라우드란?
# 워드 클라우드(Word Cloud)는 말 그대로 단어를 구름처럼 표현하는 방법이다.
# 텍스트 마이닝 결과를 표현하는 가장 대표적인 방법 중 하나로 많은 키워드 중에서 
# 가장 많이 도출된 단어를 크기와 색상으로 강조해 시각화시킨 것이다.

# <div>
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Web_2.0_Map.svg/330px-Web_2.0_Map.svg.png" width="500"/>
#     <center>워드클라우드(출처 : wikipedia)</center>
# </div>

#%%

# 패키지 설치

# JDK 설치하기
# 우선 jdk 설치가 필요하며 최신버전 설치
# https://www.oracle.com/java/technologies/downloads/

# 패키지 설치하기
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#jpype 
# 여기서 cp38(파이썬 3.8.X) 버전에 맞는 JPype 1.1.2를 
# c:\에 다운받아 설치
# get_ipython().system('pip install C:\\JPype1-1.1.2-cp38-cp38-win_amd64.whl')

# tweepy 오류로 인해 특정버전 설치
# get_ipython().system('pip install tweepy==3.10.0')

# 패키지 설치
# get_ipython().system('pip install wordcloud')
# get_ipython().system('pip install counter')
# get_ipython().system('pip install konlpy')

#%%

###############################################################################
# 텍스트 데이터 가공하기
###############################################################################

#%%

# 워드클라우드 구현을 위한 함수 불러오기
from wordcloud import WordCloud
from collections import Counter
from konlpy.tag import Hannanum


#%%

# 한글파일이기 때문에 cp949 에러가 나므로 encoding 옵션을 추가
txt = open('dataset/ch8.txt','rt', encoding='UTF-8').read()

#%%

# 텍스트 확인
txt

#%%

# txt 문서에서 명사만 추출하기
n = Hannanum().nouns(txt)


#%%

# 명사 15개만 확인
n[0:15]


#%%

# 특정값의 인덱스 찾기
n.index('재구')


#%%

# 인덱스를 이용해 특정값 변경
n[14] = '재구매'


#%%

# 제대로 변경되었는지 확인
n[14]


#%%

# 글자수가 2개 이상인 단어만 필터링
n2 = [item for item in n if len(item) >= 2]


#%%

# 명사별 빈도 추출
cnt = Counter(n2)



#%%

# 빈도수가 많은 단어 순서대로 10개만 표시
cnt.most_common()[0:10]

#%%

###############################################################################
# 워드 클라우드 그리기
###############################################################################

# 워드클라우드용 폰트 설정(맑은고딕)
wcs = WordCloud(font_path = 'C:/Windows/Fonts/malgun.ttf', background_color = 'white')

#%%

# 단어별 빈도로 워드클라우드 생성
cloud = wcs.generate_from_frequencies(cnt)

#%%

# 워드클라우드 그리기
import matplotlib.pyplot as plt
plt.figure(figsize = (10,8))
plt.axis('off')
plt.imshow(cloud)
plt.show


#%%
###############################################################################

# [분석스토리]  
# 첫 판매된 닭의 고객 리뷰 30건을 워드 클라우드를 통해 분석해 봤다. 
# 다행히도 많은 분들께서 “만족”한 것 같아 기분이 좋았습니다.
# 하지만 부정적인 단어도 많이 보였기 때문에 정말로 만족한 고객분들이 많았던 건지 의심이 가기 시작했다.
# 그래서 단어가 아닌 문장 단위로 긍정을 나타내는지, 부정을 나타내는지, 
# 아니면 중립을 나타내는지 감성 분석을 실시해 보기로 했다.

# 고객들은 정말로 만족했을까? (감성 분석)

# 감성 분석이란?
# 감성 분석(Sentiment Analysis)은 텍스트 속에서 감성, 의견 등과 같은 
# 주관적인 정보를 체계적으로 식별, 추출, 정량화하는 기술이다.
# 감성 분석은 주로 리뷰 및 설문조사 응답, SNS 결과 등을 분석하는 데 사용된다.
# 분석한 결과는 주로 파이 차트를 이용해 긍정, 부정의 비율이 몇 퍼센트인지 표시해 주는 것으로 나타낸다.

#%%

###############################################################################
# 1. 감성 사전 준비하기
###############################################################################

#%%

# 이미 만들어진 리뷰 감성 분석 전용 사전 불러옴
import pandas as pd
sentdic = pd.read_csv('dataset/review_dict.txt', sep = '\t', encoding = 'utf-8', header = None)
sentdic.columns = ['word','score']

#%%

# 감성사전 확인
sentdic.head(10)

#%%

###############################################################################
# 2. 텍스트 데이터 가공하기
###############################################################################

# DTM(Document Term Matrix)
# 문장을 카운트 벡터화
# 카운트 벡터화는 전체 문장을 띄어쓰기 기준으로 단어(term)로 쪼갠 대음 열(column)로 두고
# 각 문자은 행(row)을 둬서 해당 단어가 나타나는 횟수를 행렬로 표현 하는 것
# 문장은 문서(document)를 의미하며 결과적으로 문서별 단어를 카운트해서 중요성을 파악하는 형태로 만드는 것

#%%

# 리뷰 한줄씩 리스트로 불러오기
txt_list = open('dataset/ch8.txt','rt', encoding='UTF-8').readlines()

#%%

# 데이터 5문장만 확인하기
txt_list[0:5]

#%%

# 문장 벡터화를 위함 함수 불러오기
from sklearn.feature_extraction.text import CountVectorizer
vector = CountVectorizer()

#%%

# 문장을 벡터화(띄어쓰기 기준으로 단어 분할)
bow = vector.fit_transform(txt_list)

#%%

# 단어를 열이름으로 지정하기 위해 저장
# term = vector.get_feature_names()
term = vector.get_feature_names_out()

#%%

import pandas as pd

# Document Term Matrix 타입으로 데이터프레임 생성
dtm = pd.DataFrame.sparse.from_spmatrix(data = bow, columns = term)

#%%

# 형태 확인(30개의 문장, 191개의 단어)
dtm.shape

#%%

dtm

#%%

###############################################################################
# 3. 감성 분석
###############################################################################

#%%

# 한 줄씩 0~190번째 열까지 1이상인 열을 찾으면 해당 열이름을 가져와 감성사전에 있는지 확인 후
# 감성사전에 있으면 해당되는 점수를 가져와서 갯수와 곱한 다음 리스트 변수 d에 저장, 아니면 0을 저장
# 총 30개의 행(문장)을 모두 마치면 각 행별 리스트의 합이 ds에 하나씩 추가되서 저장됨

#%%

ds = []
for i in range(dtm.shape[0]):    
    d = []
    for j in range(dtm.shape[1]):
        if dtm.iloc[i,j]>=1:
            if sentdic.loc[sentdic.word == dtm.columns[j],'score'].empty != True:
                d.append(sentdic.loc[sentdic.word == dtm.columns[j],'score'].values[0]*dtm.iloc[i,j])
            else:
                d.append(0)
        else:
            d.append(0)
    s = sum(d)
    ds.append(s)

#%%

# 각 행(문서)별 합계 5개만 확인
ds[0:5]

#%%

# 리뷰 리스트와 리뷰별 점수합계 결과를 데이터프레임으로 만들기
res = pd.DataFrame(list(zip(txt_list, ds)), columns = ['doc','score'])

#%%

# 리뷰별 점수결과 6개만 확인
res.head(6)

#%%

# 리뷰점수가 0보다 크면 긍정, 0이면 중립, 그외면 부정으로 판정하는 리스트 만들기
pn = []
for row in res['score']:
    if row > 0:
        pn.append('Positive')
    elif row == 0:
        pn.append('Neutral')
    else:
        pn.append('Negative')


#%%

# 리뷰점수 판정결과 데이터프레임에 추가
res['pn'] = pn


#%%

# 리뷰결과 데이터프레임 6개만 확인
res.head(6)

#%%

###############################################################################
# 4. 결과 시각화
###############################################################################

#%%

# 판정결과를 갯수기준으로 그룹화
res_g = res.groupby([pn]).count()

#%%

# 그룹화 결과 확인
res_g

#%%

# 파이차트 그리기위해 비율 계산
pct = res_g['pn']/res_g['pn'].sum()

#%%

# 파이차트 그리기
plt.figure(figsize=(10, 10))
plt.pie(pct, labels = res_g.index, autopct='%.1f%%', startangle = 0, counterclock = False)
plt.show()

#%%

# THE END