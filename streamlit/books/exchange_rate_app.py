# 환율 데이터를 가져오는 웹 앱

import streamlit as st
import pandas as pd
import datetime
import time
import matplotlib.pyplot as plt
import matplotlib
from io import BytesIO

# -----------------------------------------------------------------------------
# 날짜별 환율 데이터를 반환하는 함수
# - 입력 인수: currency_code(통화코드), last_page_num(페이지 수)
# - 반환: 환율 데이터
# -----------------------------------------------------------------------------
def get_exchange_rate_data(currency_code, last_page_num):
    base_url = "https://finance.naver.com/marketindex/exchangeDailyQuote.nhn"
    df = pd.DataFrame()
    
    for page_num in range(1, last_page_num+1):
        url = f"{base_url}?marketindexCd={currency_code}&page={page_num}"
        dfs = pd.read_html(url, header=1)
        
        # 통화 코드가 잘못 지정됐거나 마지막 페이지의 경우 for 문을 빠져나옴
        if dfs[0].empty:
            if (page_num==1):
                print(f"통화 코드({currency_code})가 잘못 지정됐습니다.")
            else:
                print(f"{page_num}가 마지막 페이지입니다.")
            break
            
        # page별로 가져온 DataFrame 데이터 연결
        df = pd.concat([df, dfs[0]], ignore_index=True)
        time.sleep(0.1) # 0.1초간 멈춤
        
    return df
# -----------------------------------------------------------------------------
  
st.title("환율 정보를 가져오는 웹 앱")

# 사이드바의 폭을 조절. {width:250px;}로 지정하면 폭을 250픽셀로 지정
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{width:250px;}
    </style>
    """, unsafe_allow_html=True
)

currency_name_symbols = {"미국 달러":"USD", "유럽연합 유로":"EUR",
                         "일본 엔(100)":"JPY", "중국 위안":"CNY"}
currency_name = st.sidebar.selectbox('통화 선택', currency_name_symbols.keys())

clicked = st.sidebar.button("환율 데이터 가져오기")

if(clicked==True):

    currency_symbol = currency_name_symbols[currency_name] # 환율 심볼 선택
    currency_code = f"FX_{currency_symbol}KRW"

    last_page_num = 20 # 네이버 금융에서 가져올 최대 페이지 번호 지정
    
    # 지정한 환율 코드를 이용해 환율 데이터 가져오기
    df_exchange_rate = get_exchange_rate_data(currency_code, last_page_num)
    
    # 원하는 열만 선택
    df_exchange_rate = df_exchange_rate[['날짜', '매매기준율','사실 때',
                                         '파실 때', '보내실 때', '받으실 때']]
    
    # 최신 데이터와 과거 데이터의 순서를 바꿔 df_exchange_rate2에 할당
    df_exchange_rate2 = df_exchange_rate[::-1].reset_index(drop=True)

    # df_exchange_rate2의 index를 날짜 열의 데이터로 변경
    df_exchange_rate2 = df_exchange_rate2.set_index('날짜')

    # df_exchange_rate2의 index를 datetime 형식으로 변환
    df_exchange_rate2.index = pd.to_datetime(df_exchange_rate2.index,
                                             format='%Y-%m-%d')

    # 1) 환율 데이터 표시
    st.subheader(f"[{currency_name}] 환율 데이터")
    st.dataframe(df_exchange_rate.head())  # 환율 데이터 표시(앞의 일부만 표시)
    
    # 2) 차트 그리기
    # matplotlib을 이용한 그래프에 한글을 표시하기 위한 설정
    matplotlib.rcParams['font.family'] = 'Malgun Gothic'
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 선 그래프 그리기 (df_exchange_rate2 이용)
    ax = df_exchange_rate2['매매기준율'].plot(grid=True, figsize=(15, 5))
    ax.set_title("환율(매매기준율) 그래프", fontsize=30) # 그래프 제목을 지정
    ax.set_xlabel("기간", fontsize=20)                   # x축 라벨을 지정
    ax.set_ylabel(f"원화/{currency_name}", fontsize=20)  # y축 라벨을 지정
    plt.xticks(fontsize=15)             # X축 눈금값의 폰트 크기 지정
    plt.yticks(fontsize=15)             # Y축 눈금값의 폰트 크기 지정
    fig = ax.get_figure()               # fig 객체 가져오기
    st.pyplot(fig)                      # 스트림릿 웹 앱에 그래프 그리기
    
    # 3) 파일 다운로드
    st.markdown("**환율 데이터 파일 다운로드**")
    # DataFrame 데이터를 CSV 데이터(csv_data)로 변환
    csv_data = df_exchange_rate.to_csv()

    # DataFrame 데이터를 엑셀 데이터(excel_data)로 변환
    excel_data = BytesIO() # 메모리 버퍼에 바이너리 객체 생성
    df_exchange_rate.to_excel(excel_data) # 엑셀 형식으로 버퍼에 쓰기

    columns = st.columns(2) # 2개의 세로단으로 구성
    with columns[0]:
        st.download_button("CSV 파일 다운로드", csv_data,
                           file_name='exchange_rate_data.csv')
    with columns[1]:
        st.download_button("엑셀 파일 다운로드", excel_data,
                           file_name='exchange_rate_data.xlsx')
