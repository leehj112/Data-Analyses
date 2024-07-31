# 차트 표시 예제1

import streamlit as st
import pandas as pd

data1 = [ -2,   5,  -3,  -3,   9,  -4,  -7,  -9,   2,   3]
data2 = [  3,   4,  -4,  -2,  -3,  -2,   0,   7,  -6,   6]
data3 = [-10,   2,   8,   6,  -7,  -1,  -4,  -1,   4,   5]

dict_data = {"data1":data1, "data2":data2, "data3":data3}
df = pd.DataFrame(dict_data) # DataFrame

st.title("스트림릿에서 차트 그리기")

st.subheader("꺾은선형 차트: st.line_chart(df) 이용")
st.line_chart(df, height=170) # 높이 지정

st.subheader("영역형 차트: st.area_chart(df) 이용")
st.area_chart(df, height=170) # 높이 지정

st.subheader("세로 막대형 차트: st.bar_chart(df) 이용")
st.bar_chart(df, height=170) # 높이 지정
