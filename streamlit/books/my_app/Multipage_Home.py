# 멀티페이지 웹 앱

import streamlit as st

st.title("경제 정보를 가져오는 웹 앱")

st.subheader("사이드바에서 페이지를 선택하세요.")
st.subheader("- Multipage Home: 경제 정보 홈 페이지")
st.subheader("- Stock Info: 주식 정보 페이지")
st.subheader("- Exchange Rate: 환율 정보 페이지")
st.subheader("- Land Info: 부동산 정보 페이지")

# 사이드바의 폭을 조절. {width:250px;} 으로 지정하면 폭을 250픽셀로 지정함
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{width:250px;}
    </style>
    """, unsafe_allow_html=True
)
