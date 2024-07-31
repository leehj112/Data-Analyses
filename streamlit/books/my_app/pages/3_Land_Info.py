# 부동산 데이터를 가져오는 웹 앱

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 원본 DataFrame의 제목 열에 있는 문자열을 분리해 
# 전국, 서울, 수도권의 매매가 변화율 열이 있는 DataFrame 반환하는 함수
#-----------------------------------------------------------------------------------
def split_title_to_rates(df_org):
    df_new = df_org.copy()

    df_temp = df_new['제목'].str.replace('%', '') # 제목 문자열에서 % 제거
    df_temp = df_temp.str.replace('보합', '0')    # 제목 문자열에서 보합을 0으로 바꿈
    df_temp = df_temp.str.replace('보합세', '0')  # 제목 문자열에서 보합세를 0으로 바꿈
    
    regions = ['전국', '서울', '수도권']    
    for region in regions:
        df_temp = df_temp.str.replace(region, '') # 문자열에서 전국, 서울, 수도권 제거

    df_temp = df_temp.str.split(']', expand=True) # ]를 기준으로 열 분리
    df_temp = df_temp[1].str.split(',', expand=True) # ,를 기준으로 열 분리
    
    df_temp = df_temp.astype(float)
    
    df_new[regions] = df_temp # 전국, 서울, 수도권 순서대로 DataFrame 데이터에 할당

    return df_new[['등록일'] + regions + ['번호']] # DataFrame에서 필요한 열만 반환
#-----------------------------------------------------------------------------------

st.title("부동산 정보를 가져오는 웹 앱")

# 사이드바의 폭을 조절. {width:250px;} 으로 지정하면 폭을 250픽셀로 지정함
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{width:250px;}
    </style>
    """, unsafe_allow_html=True
)

# 선택을 위한 체크박스를 생성
checked_1 = st.sidebar.checkbox('전국')
checked_2 = st.sidebar.checkbox('서울')
checked_3 = st.sidebar.checkbox('수도권')

clicked = st.sidebar.button("부동산 데이터 가져오기") # 버튼 생성
      
if(clicked==True):    
    st.subheader("아파트의 매매가 변화율 데이터")
    
    base_url = "https://land.naver.com/news/trendReport.naver"

    df_rates = pd.DataFrame() # 전체 데이터가 담길 DataFrame 데이터
    last_page_num = 2 # 가져올 데이터의 마지막 페이지 

    for page_num in range(1, last_page_num+1):

        url = f"{base_url}?page={page_num}"
        dfs = pd.read_html(url)

        df_page = dfs[0] # 리스트의 첫 번째 항목에 동향 보고서 제목 데이터가 있음
        df_rate = split_title_to_rates(df_page)

        # 세로 방향으로 연결 (기존 index를 무시)
        df_rates = pd.concat([df_rates, df_rate], ignore_index=True) 

    # 최신 데이터와 과거 데이터의 순서를 바꿈. index도 초기화함  
    df_rates_for_chart = df_rates[::-1].reset_index(drop=True)
    
    selected_regions = []
    
    if(checked_1==True):
        selected_regions.append("전국")
    if(checked_2==True):
        selected_regions.append("서울")   
    if(checked_3==True):
        selected_regions.append("수도권")

    if(selected_regions == []):
        st.subheader("지역을 선택하세요.")
    else:
        # 1) 매매가 변화율 표시. 환율 데이터를 앞의 일부만 표시
        st.dataframe(df_rates[['등록일']+selected_regions].head()) 
    
        # 2) 차트 그리기
        # matplotlib을 이용한 그래프에 한글을 표시하기 위한 설정
        matplotlib.rcParams['font.family'] = 'Malgun Gothic'
        matplotlib.rcParams['axes.unicode_minus'] = False

        # 선 그래프 그리기 (df_exchange_rate2 이용)        
        ax = df_rates_for_chart.plot(x='등록일', y=selected_regions, figsize=(15, 6),
                                     style = '-o', grid=True) # 그래프 그리기
        
        ax.set_title("아파트 매매가 변화율", fontsize=30) # 그래프 제목을 지정
        ax.set_xlabel("날짜", fontsize=20)                   # x축 라벨을 지정
        ax.set_ylabel("변화율(%)", fontsize=20)  # y축 라벨을 지정
        plt.xticks(fontsize=15)             # X축 눈금값의 폰트 크기 지정
        plt.yticks(fontsize=15)             # Y축 눈금값의 폰트 크기 지정    
        fig = ax.get_figure()               # fig 객체 가져오기    
        st.pyplot(fig)                      # 스트림릿 웹 앱에 그래프 그리기
