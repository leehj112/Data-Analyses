# 세로단(컬럼) 분할 사용 예제

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np

st.title("스트림릿에서 화면 분할 사용 예")

# 1) 2개로 세로단 분할 (예제 1)
folder = './books/data/ch09/'    # 폴더 지정
file = folder + '공장별_생산현황2.csv'   # 데이터 파일 지정
df = pd.read_csv(file, index_col='year') # CSV 파일을 DataFrame 데이터로 읽기

[col1, col2] = st.columns(2) # 너비가 같은 2개의 세로단으로 구성

with col1: # 첫 번째 세로단(컬럼)
    st.subheader("DataFrame 데이터")
    st.dataframe(df)  # DataFrame 데이터 표시

with col2: # 두 번째 세로단(컬럼)
    st.subheader("꺾은 선 차트")
    st.line_chart(df) # 꺾은 선 차트 표시

# 2) 3개로 세로단 분할 (예제 2)
columns = st.columns([1.1, 1.0, 0.9]) # 너비가 다른 3개의 세로단으로 구성

image_files = ['dog.png', 'cat.png', 'bird.png'] # 이미지 파일명 리스트
image_cations = ['강아지', '고양이', '새']       # 이미지 파일명 리스트

for k in range(len(columns)):
    with columns[k]: # 세로단(컬럼) 지정
        st.subheader(image_cations[k])                    # 세로단 별로 subheader 표시
        image_local = Image.open(folder + image_files[k]) # 세로단 별로 이미지 열기
        st.image(image_local,caption=image_cations[k])    # 세로단 별로 이미지 표시
