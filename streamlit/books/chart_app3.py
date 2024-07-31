# 차트 표시 예제3

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# matplotlib을 이용한 그래프에 한글을 표시하기 위한 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# 꺾은선형 차트 데이터
folder = 'C:/myPyScraping/data/ch09/'
file = folder + '공장별_생산현황.csv'
df1 = pd.read_csv(file, index_col='year')

# 세로 막대형 차트 데이터
file = folder + '영업팀별_판매현황.xlsx'
df2 = pd.read_excel(file, index_col='월')

st.title("스트림릿에서 차트 그리기")

# 꺾은선형 차트 그리기
ax = df1.plot(grid=True, figsize=(15,5))
ax.legend(['공장 A', '공장 B', '공장 C'], fontsize=10)
ax.set_title("공장별 생산 현황", fontsize=20) # 그래프 제목을 지정
ax.set_xlabel("연도", fontsize=15)            # x축 라벨을 지정
ax.set_ylabel("생산량", fontsize=15)          # y축 라벨을 지정
fig1 = ax.get_figure()                        # fig 객체 가져오기

st.subheader("꺾은선형 차트: Matplotlib과 st.pyplot(fig) 이용")
st.pyplot(fig1) # 스트림릿 웹 앱에 그래프 그리기

# 세로 막대형 차트 그리기
ax = df2.plot.bar(grid=True, rot=0,  figsize=(15,5))
ax.set_title("영업팀별 판매현황", fontsize=20) # 그래프 제목을 지정
ax.set_xlabel("월", fontsize=15)               # x축 라벨을 지정
ax.set_ylabel("판매현황", fontsize=15)         # y축 라벨을 지정
fig2 = ax.get_figure()                         # fig 객체 가져오기

st.subheader("세로 막대형 차트: Matplotlib과 st.pyplot(fig) 이용")
st.pyplot(fig2) # 스트림릿 웹 앱에 그래프 그리기
