# 이미지 표시 사용 예제

import streamlit as st
from PIL import Image

st.title("스트림릿의 이미지 표시 사용 예")

# 1) 컴퓨터 내에 있는 이미지 파일을 열어서 표시
st.subheader("1. 컴퓨터 내의 이미지 파일을 표시")
image_file = 'C:/myPyScraping/data/ch09/avenue.jpg' # 이미지 파일 경로
image_local = Image.open(image_file)                # PIL 라이브러리의 Image.open() 함수로 이미지 파일 열기
st.image(image_local, width=350, caption='컴퓨터 내의 이미지 파일을 열어서 표시한 이미지') # 이미지 표시

# 2) 웹상에 있는 이미지의 주소(URL)를 이용해 이미지 표시
st.subheader("2. 웹상에 있는 이미지 파일을 표시")
image_url = "https://cdn.pixabay.com/photo/2015/05/04/10/16/vegetables-752153_960_720.jpg" # 이미지 URL
st.image(image_url, width=350, caption='웹상에 있는 이미지의 주소(URL)를 지정해 표시한 이미지') # 이미지 표시
