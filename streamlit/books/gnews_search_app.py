# 구글 뉴스에서 기사를 검색하는 웹 앱

from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import feedparser

# RSS 피드 제공 일시를 한국 날짜와 시간으로 변경하는 함수
def get_local_datetime(rss_datetime):    
    # 전체 값 중에서 날짜와 시간만 문자열로 추출 
    date_time_str = ' '.join(rss_datetime.split()[1:5])
    
    # 문자열의 각 자리에 의미를 부여해 datetime 객체로 변경 
    date_time_GMT = datetime.strptime(date_time_str, '%d %b %Y %H:%M:%S') 
    
    # GMT에 9시간을 더해 한국 시간대로 변경
    date_time_KST = date_time_GMT + timedelta(hours=9) 
    
    return date_time_KST # 변경된 시간대의 날짜와 시각 반환 
#---------------------------------------------------------

# 구글 뉴스 RSS 피드에서 검색 결과를 가져와 DataFrame 데이터로 반환하는 함수 
def get_gnews(query):
    # RSS 서비스 주소
    rss_url = f'https://news.google.com/rss/search?q={query}&&hl=ko&gl=KR&ceid=KR:ko' 
    rss_news = feedparser.parse(rss_url) # RSS 형식의 데이터를 파싱
    
    title = rss_news['feed']['title']
    updated = rss_news['feed']['updated']
    updated_KST = get_local_datetime(updated) # 한국 날짜와 시각으로 변경   
    
    df_gnews = pd.DataFrame(rss_news.entries) # 구글 뉴스 아이템을 판다스 DataFrame으로 변환

    selected_columns = ['title', 'published', 'link'] # 관심있는 열만 선택
    df_gnews2 = df_gnews[selected_columns].copy()     # 선택한 열만 다른 DataFrame으로 복사

    # published 열의 작성 일시를 한국 시간대로 변경
    df_gnews2['published'] = df_gnews2['published'].apply(get_local_datetime) 

    df_gnews2.columns = ['제목', '제공 일시', '링크'] # 열 이름 변경
    
    return title, updated_KST, df_gnews2
#---------------------------------------------------------

# 구글 뉴스 검색 결과를 Table로 정리한 HTML 코드를 반환하는 함수
def create_gnews_html_code(title, updated_KST, df):
    # DataFrame 데이터를 HTML 코드로 변환 (justify='center' 옵션을 이용해 열 제목을 중간에 배치)
    html_table = df.to_html(justify='center', escape=False, render_links=True) 

    # HTML 기본 구조를 갖는 HTML 코드
    html_code = '''
    <!DOCTYPE html>
    <html>
      <head>
        <title>구글 뉴스 검색</title>
      </head>
      <body>
        <h1>{0}</h1>
        <h3> *검색 날짜 및 시각: {1}</h3>
        {2}
      </body>
    </html>    
    '''.format(title, updated_KST, html_table)
    
    return html_code
# -----------------------------------

st.title("구글 뉴스 기사를 검색하는 웹 앱")

query = st.text_input('검색어 입력', value="메타버스")

# 구글 뉴스 검색 결과 가져오기
[title_gnews, updated_KST_gnews, df_gnews] = get_gnews(query)

# 웹 앱에 표시할 HTML 테이블 생성 (DataFrame 데이터 중 처음 일부만 HTML 테이블로 생성)
html_table = df_gnews.head().to_html(justify='center', escape=False, render_links=True)

# HTML 파일 다운로드를 위한 HTML code 생성
gnews_html_code = create_gnews_html_code(title_gnews, updated_KST_gnews, df_gnews)

st.markdown("**기사 검색 결과**")
st.write(html_table, unsafe_allow_html=True) # HTML 테이블 표시
st.markdown("") # 빈 줄 생성

columns = st.columns(3) # 3개의 세로단으로 구성 (2개의 세로단에만 필요한 내용 표시)
with columns[0]:        
    st.markdown("**HTML 파일 다운로드**")
with columns[1]:
    st.download_button("다운로드", gnews_html_code, file_name='gnews_search_results.html')
