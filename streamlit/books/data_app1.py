# 데이터 표시 예제1

import streamlit as st
import pandas as pd

# CSV 파일 경로
folder = 'C:/myPyScraping/data/ch09/' # 폴더 경로를 지정
csv_file = folder + 'korea_rain1.csv' # 파일 경로를 지정

# CSV 파일을 읽어와서 DataFrame 데이터 생성
df = pd.read_csv(csv_file, encoding="utf-8")

st.title("스트림릿에서 데이터 표시 (1/2)")

st.subheader("st.dataframe() 이용")
st.dataframe(df)

st.subheader("st.table() 이용")
st.table(df)
