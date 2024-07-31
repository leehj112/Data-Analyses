# 숫자 입력의 사용 예제

import streamlit as st

st.title("스트림릿의 숫자 입력 사용 예")

number1 = st.number_input('20 이상의 두 자리 숫자를 입력하세요', min_value=20, max_value=99)
st.write('**입력한 숫자**', number1)

height = st.number_input('키(cm)를 입력하세요', min_value=1.0, value=170.0, step=0.1)
weight = st.number_input('몸무게(kg)를 입력하세요', min_value=1.0, value=65.5)
BMI = weight/((height/100)**2)
st.write('**신체질량지수(BMI):**', BMI )
