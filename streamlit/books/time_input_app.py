# 시각 입력의 사용 예제

import streamlit as st
import datetime

st.title("스트림릿의 시각 입력 사용 예")

alarm_time = st.time_input("알람 시각을 설정하세요", value=datetime.time(7, 30)) # 오전 7시 30분
st.write("**알람 설정 시각:** ", alarm_time)

work_start_time = st.time_input("업무 시작 시각을 설정하세요", value=datetime.time(9)) # 오전 9시
st.write("**업무 시작 시각:** ", work_start_time)
