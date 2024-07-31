# 날짜 입력의 사용 예제

import streamlit as st
import datetime

st.title("스트림릿의 날짜 입력 사용 예")

# 날짜 지정
birthday = st.date_input("1. 당신의 생일은 언제입니까?", 
                          value=datetime.date(2000, 1, 1))
st.write("당신의 생일: ", birthday)

# 날짜의 범위 지정
date_range = st.date_input("2. 시작과 끝 날짜를 선택해 주세요", 
                           value=[datetime.date(2022, 1, 10), datetime.date(2022, 2, 1)],
                           min_value=datetime.date(2022, 1, 5),
                           max_value=datetime.date(2022, 2, 20))
st.write("당신이 선택한 날짜의 범위: ", date_range)
