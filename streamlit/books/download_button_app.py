# 다운로드 버튼 사용 예제

import streamlit as st
import pandas as pd
from io import BytesIO

KTX_data = {'경부선 KTX': [43621, 41702, 41266, 32427],
            '호남선 KTX': [6626, 8675, 10622, 9228],
            '경전선 KTX': [4424, 4606, 4984, 5570],
            '전라선 KTX': [2244, 3146, 3945, 5766]}
col_list = ['경부선 KTX','호남선 KTX','경전선 KTX','전라선 KTX']
index_list = ['2014', '2015', '2016', '2017']

df = pd.DataFrame(KTX_data, columns = col_list, index = index_list)

# DataFrame 데이터를 CSV 데이터(csv_data)로 변환
csv_data = df.to_csv()

# DataFrame 데이터를 엑셀 데이터(excel_data)로 변환
excel_data = BytesIO() # 메모리 버퍼에 바이너리 객체 생성
df.to_excel(excel_data) # DataFrame 데이터를 엑셀 형식으로 버퍼에 쓰기

# 스트림릿 화면 구성
st.title("스트림릿의 다운로드 버튼 사용 예")

st.subheader("DataFrame 데이터")
st.dataframe(df)
             
st.subheader("CSV 파일로 다운로드")
st.download_button("CSV 파일 다운로드", csv_data, file_name='KTX_users.csv')

st.subheader("엑셀 파일로 다운로드")
st.download_button("엑셀 파일 다운로드", excel_data, file_name='KTX_users.xlsx')
