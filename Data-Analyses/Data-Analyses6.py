# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:30:15 2024

@author: leehj
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'

#%% 함수 생성
# 데이터프레임(df)에서 열(col) 값(value)에 해당하는 데이터만 추출
def filter_df(df, col, value) :
    if value == '전체' :
        data = df.copy()
    else :
        data = df.copy()
        data = data[data[col] == value]
    return data

# 데이터프레임에서 지역에 따른 인구(합계)/면적(합계)/밀도(인구/면적) 계산
#   - 해당 지역의 상위 레벨에서 데이터프레임을 만들어서 해당 지역 값 추출
def density(df, dept) :
    data_table = df[[dept, '인구수', '면적(km2)']]
    data_table = data_table.groupby(dept).sum()
    data_table['인구밀도'] = round(data_table['인구수'] / data_table['면적(km2)'], 2)
    data_table.loc['평균'] = data_table.mean(axis =0)
    data_table.loc['합계'] = data_table[:-1].sum(axis=0)
    data_table.iloc[:, 1:3] = round(data_table.iloc[:, 1:3], 2)
    return data_table

# 데이터프레임에서 선택 업종의 하위업종에 따른 매장개수 계산
#   - 열 추가 : 비율 = 하위업종 개수 / 해당업종 개수 
def store_cnt(df, dept) :
    data_series = df[[dept]]
    data_series = data_series.groupby(dept).value_counts()
    
    total = data_series.sum()
    data_table = pd.DataFrame({'매장개수': data_series, '비율(%)': round((data_series / total)*100,2)})
    
    return data_table

#%% [사이드바]

#%% 지역 선택
df_density = pd.read_csv('./행정구역별 인구수_면적.csv')
# - 하위 : 시 > 군구 > 행정동
st.sidebar.title('지역 선택')
df_col = df_density.copy()
# '시' 셀렉트박스
si_op = np.sort(df_col['시'].unique())
si_op = np.append(si_op, '전체')
si_op_choice = st.sidebar.selectbox(label = '시군구', options=si_op, index=int(np.where(si_op == "전체")[0][0]))
# - '시' 선택에 따라 데이터 추출
if si_op_choice == '전체' :
    df_col = df_col.copy()
else : 
    df_col = df_col[ df_col['시'] == si_op_choice ]
# '군구' 셀렉트박스  
gu_op = np.sort(df_col['군구'].unique())
gu_op = np.append(gu_op, '전체')
gu_op_choice = st.sidebar.selectbox(label = '군구', options=gu_op, index=int(np.where(gu_op == "전체")[0][0]))
# - '군구' 선택에 따라 데이터 추출
if gu_op_choice == '전체' :
    df_col = df_col.copy()
else : 
    df_col = df_col[ df_col['군구'] == gu_op_choice ]
# - '군구' 선택 -> '시' 값 부여
    si_op_choice = (df_col[df_col['군구'] == gu_op_choice]['시'].unique())[0]
# '행정동' 셀렉트박스 
dong_op = np.sort(df_col['행정동명'].unique())
dong_op = np.append(dong_op, '전체')
dong_op_choice = st.sidebar.selectbox(label = '행정동', options=dong_op, index=int(np.where(dong_op == "전체")[0][0]))
# - '행정동' 선택 -> '시', '군구' 값 부여
if dong_op_choice != '전체' :
    si_op_choice = (df_col[df_col['행정동명'] == dong_op_choice]['시'].unique())[0]
    gu_op_choice = (df_col[df_col['행정동명'] == dong_op_choice]['군구'].unique())[0]
    
# 지역 선택에 따른 변수값 지정
region_op = {'시' : si_op_choice, '군구' : gu_op_choice, '행정동명' : dong_op_choice}

if dong_op_choice != '전체' : 
    idx = 2 
    title_region = f'{si_op_choice}-{gu_op_choice}-{dong_op_choice}'
elif gu_op_choice != '전체' : 
    idx = 1
    title_region = f'{si_op_choice}-{gu_op_choice}'
else :
    idx = 0
    if si_op_choice != '전체' :
        title_region = f'{si_op_choice}'
    else :
        title_region = '수원시 & 화성시'
        
def region_items(idx) :
    key, value = list(region_op.items())[idx]
    return key, value     
        
#%% 업종 선택
# - 하위 : 대분류명 > 중분류명 > 소분류명 
df_store = pd.read_csv('./상가(상권)정보_시군구통일.csv')
df_store_copy = df_store[['시','군구','행정동명', '상권업종대분류명', '상권업종중분류명', '상권업종소분류명']]

df_col_s = df_store_copy.copy()    

# '대분류명' 셀렉트박스
lc_op = np.sort(df_col_s['상권업종대분류명'].unique())
lc_op = np.append(lc_op, '전체')
lc_op_choice = st.sidebar.selectbox(label = '상권업종대분류', options=lc_op, index=int(np.where(lc_op == "전체")[0][0]))
# - '대분류명' 선택에 따라 데이터 추출
if lc_op_choice == '전체' :
    df_col_s = df_col_s.copy()
else : 
    df_col_s = df_col_s[ df_col_s['상권업종대분류명'] == lc_op_choice ]
    
# '중분류명' 셀렉트박스
mc_op = np.sort(df_col_s['상권업종중분류명'].unique())
mc_op = np.append(mc_op, '전체')
mc_op_choice = st.sidebar.selectbox(label = '상권업종중분류', options=mc_op, index=int(np.where(mc_op == "전체")[0][0]))
# - '중분류명' 선택에 따라 데이터 추출
if mc_op_choice == '전체' :
    df_col_s = df_col_s.copy()
else : 
    df_col_s = df_col_s[ df_col_s['상권업종중분류명'] == mc_op_choice ]
# - '중분류명' 선택 -> '대분류명' 값 부여
    lc_op_choice = (df_col_s[df_col_s['상권업종중분류명'] == mc_op_choice]['상권업종대분류명'].unique())[0] 
    
# '소분류명' 셀렉트박스  
sc_op = np.sort(df_col_s['상권업종소분류명'].unique())
sc_op = np.append(sc_op, '전체')
sc_op_choice = st.sidebar.selectbox(label = '상권업종중분류', options=sc_op, index=int(np.where(sc_op == "전체")[0][0]))
# - '소분류명' 선택 -> '대분류명', '중분류명' 값 부여
if sc_op_choice != '전체' :
    lc_op_choice = (df_col_s[df_col_s['상권업종소분류명'] == sc_op_choice]['상권업종대분류명'].unique())[0]
    mc_op_choice = (df_col_s[df_col_s['상권업종소분류명'] == sc_op_choice]['상권업종중분류명'].unique())[0]
    
# 업종 선택에 따른 변수값 지정
sector_op = {'상권업종대분류명' : lc_op_choice, '상권업종중분류명' : mc_op_choice, '상권업종소분류명' : sc_op_choice}

if sc_op_choice != '전체' : 
    idx_s = 2 
    title_sector = f'{sc_op_choice}'
elif mc_op_choice != '전체' : 
    idx_s = 1  
    title_sector = f'{mc_op_choice}'
else :
    idx_s = 0
    if lc_op_choice != '전체' :
        title_sector = f'{lc_op_choice}'
    else :
        title_sector = '전체'
        
def sector_items(idx_s) :
    key, value = list(sector_op.items())[idx_s]
    return key, value     

    
#%%    
if st.sidebar.button('데이터 조회') :      
    
    # [함수를 통한 인구, 면적, 인구밀도 데이터 연산]
    # 선택값에 따른 인구, 면적, 인구밀도
    if si_op_choice == '전체' :
        fil_df = filter_df(df_density, region_items(idx)[0], region_items(idx)[1])            
        data_den = density(fil_df, region_items(idx)[0])   
        
        total_df = data_den.sum(axis=0)
        total_p = total_df.loc['인구수']
        total_a = total_df.loc['면적(km2)']
        total_pa = total_p / total_a
    else :
        fil_df = filter_df(df_density, region_items(idx-1)[0], region_items(idx-1)[1])            
        data_den = density(fil_df, region_items(idx)[0])   
        
        total_p = data_den.loc[region_items(idx)[1], '인구수']
        total_a = data_den.loc[region_items(idx)[1], '면적(km2)']
        total_pa = data_den.loc[region_items(idx)[1], '인구밀도']
    
    # [인구 그래프용 데이터 호출]
    people_cnt = pd.read_csv('./행정구역별 연령별_성별_인구수.csv', index_col=0)
    
    if si_op_choice != '전체' :
        people_cnt = filter_df(people_cnt, region_items(idx)[0], region_items(idx)[1])
    
    age_list = list(people_cnt.columns)[3:9+1]
    age_color = ['gainsboro','salmon','darkorange','darkkhaki','darkseagreen','slategray','silver']
    
    sex_list = list(people_cnt.columns)[10:12]        
    sex_color = ['skyblue', 'pink']
    
    # [업종 데이터에서 지역 필터링]
    if si_op_choice != '전체' :
        fil_region = filter_df(df_store, region_items(idx)[0], region_items(idx)[1])
    else :
        fil_region = df_store.copy()
    
    # [해당 업종 매장 수]     
    store = filter_df(fil_region, sector_items(idx_s)[0], sector_items(idx_s)[1])  
    
    # [하위레벨 업종 매장 수/인구밀도]
    if lc_op_choice == '전체' :
        data_cnt = store_cnt(fil_region, sector_items(idx_s)[0])
    else :
        if idx_s == 2 :
            group_df = filter_df(fil_region, sector_items(idx_s-1)[0], sector_items(idx_s-1)[1])
            data_cnt = store_cnt(group_df, sector_items(idx_s)[0])         
        else :
            group_df = filter_df(fil_region, sector_items(idx_s)[0], sector_items(idx_s)[1])
            data_cnt = store_cnt(group_df, sector_items(idx_s+1)[0])  
        
    data_cnt['매장/인구밀도'] = round((data_cnt['매장개수'] / total_pa), 3)     
    
#%%     
    # [인구, 면적, 인구밀도]     
    # 선택 값의 인구, 면적, 인구밀도
    st.title(f'{title_region}의 인구 밀도 : {total_pa:,.2f}명/km2')
    
    density_all = data_den.loc['합계', '인구수'] / data_den.loc['합계', '면적(km2)']      
    if lc_op_choice != '전체' :
        if density_all < data_den.loc[region_items(idx)[1], '인구밀도'] :
            density_gap = data_den.loc[region_items(idx)[1], '인구밀도'] - density_all
            st.markdown(f'- **{region_items(idx)[1]}의 인구밀도**는 **{region_items(idx-1)[1]} 평균 인구밀도** 대비 {density_gap:,.2f} 높습니다.')
        else :
            density_gap = density_all - data_den.loc[region_items(idx)[1], '인구밀도'] 
            st.markdown(f'- **{region_items(idx)[1]}의 인구밀도**는 **{region_items(idx-1)[1]} 평균 인구밀도** 대비 {density_gap:,.2f} 낮습니다.')     
           
        per_pp = (total_p/ data_den.loc['합계', '인구수'])*100
        st.markdown(f'- **{region_items(idx)[1]}의 총 인구수**는 {total_p:,.0f}명입니다. (**{region_items(idx-1)[1]}**의 {per_pp:.1f}%)')
        
        per_area = (total_a/ data_den.loc['합계', '면적(km2)'])*100
        st.markdown(f'- **{region_items(idx)[1]}의 총 면적**은 {total_a:,.2f}km2입니다. (**{region_items(idx-1)[1]}**의 {per_area:.1f}%)')
    else :
        st.markdown(f'- **{title_region}의 총 인구수**는 {total_p:,.0f}명입니다.')
        st.markdown(f'- **{title_region}의 총 면적**은 {total_a:,.2f}km2입니다.')
        
    
    # [연령대/ 성별 비율]
    st.subheader('|인구 비율')
    columns = st.columns([1, 1])    
    with columns[0] :        
        # 해당 지역 연령대별 파이
        st.markdown('**연령별**')
        age_fig = plt.figure() 
        plt.pie(people_cnt.iloc[0, 3:10], labels = age_list, startangle=90, counterclock=False, colors = age_color, autopct = '%.1f%%')
        plt.show()
        st.pyplot(age_fig)
    with columns[1] :                    
        # 해당 지역 남녀 파이
        st.markdown('**성별**')
        sex_fig = plt.figure() 
        plt.pie(people_cnt.iloc[0, 10:12], labels = sex_list, startangle=90, counterclock=False, colors = sex_color, autopct = '%.1f%%')      
        plt.show()
        st.pyplot(sex_fig)                        
             
    # [해당 지역에 많은 업종]
    st.title(f'{title_region}에 많은 업종(Top 10)')
    rank_store = fil_region[list(sector_op.keys())]    
    rank_store = rank_store.groupby(list(sector_op.keys())).value_counts()
    
    total = rank_store.sum()
    rank_store = pd.DataFrame({'매장개수': rank_store, '비율(%)': round((rank_store / total)*100,2)})
    
    rank_store = rank_store.sort_values(by='매장개수', ascending=False).head(10)    
    st.dataframe(rank_store)     
    
#%%    
    # [업종 데이터에서 지역 필터링]
    if lc_op_choice != '전체' :
        fil_sector = filter_df(df_store, sector_items(idx)[0], sector_items(idx)[1])
    else :
        fil_sector = df_store.copy()      
        
    # [업종 매장 수]       
    st.title(f'{title_region}에 위치한 매장 개수 : {len(store):,d}') 
    
    if sc_op_choice != '전체' :
        total_s_df = data_cnt.sum(axis=0)
        total_s = total_s_df.loc['매장개수']
        s_cnt = data_cnt.loc[sector_items(idx_s)[1], '매장개수']
        per_s = (s_cnt / total_s)*100
        
        st.markdown(f'- **{sector_items(idx_s)[1]}** 매장은 **{sector_items(idx_s-1)[1]}** 매장의 {per_s:.1f}% 입니다.')
        
    elif mc_op_choice != '전체' :
        group_df = filter_df(fil_region, sector_items(idx_s-1)[0], sector_items(idx_s-1)[1])
        store_df = store_cnt(group_df, sector_items(idx_s)[0])               
        total_s_df = store_df.sum(axis=0)
        total_s = total_s_df.loc['매장개수']
        s_cnt = store_df.loc[sector_items(idx_s)[1], '매장개수']
        per_s = (s_cnt / total_s)*100
        
        st.markdown(f'- **{sector_items(idx_s)[1]}** 매장은 **{sector_items(idx_s-1)[1]}** 매장의 {per_s:.1f}% 입니다.')
        
    elif lc_op_choice != '전체' :
        group_df = filter_df(fil_region, sector_items(idx_s-1)[0], sector_items(idx_s-1)[1])
        store_df = store_cnt(group_df, sector_items(idx_s)[0])               
        total_s_df = store_df.sum(axis=0)
        total_s = total_s_df.loc['매장개수']
        s_cnt = store_df.loc[sector_items(idx_s)[1], '매장개수']
        per_s = (s_cnt / total_s)*100
        
        st.markdown(f'- **{sector_items(idx_s)[1]}** 매장은 **{sector_items(idx_s-1)[1]}** 매장의 {per_s:.1f}% 입니다.')
        
        
    
    # [매장 위치]
    st.subheader('|매장 위치')  
    df_for_map = store[['위도','경도']]
    df_for_map = df_for_map.rename(columns = {'위도':'lat','경도':'lon'})        
    st.map(df_for_map, zoom = 12)
    # [매장 리스트]
    st.subheader('|매장 리스트')
    store_fil = store[['상권업종소분류명','상호명','시','군구','행정동명','위도','경도']].reset_index(drop=True)
    st.dataframe(store_fil)
    
        
    # [해당 업종이 많은 지역]
    st.title(f'|{sector_items(idx_s)[1]} 매장이 많은 지역(Top 10)')  
    rank_region = fil_sector[list(region_op.keys())]    
    rank_region = rank_region.groupby(list(region_op.keys())).value_counts()
    
    total = rank_region.sum()
    rank_region = pd.DataFrame({'매장개수': rank_region, '비율(%)': round((rank_region / total)*100,2)})
    
    rank_region = rank_region.sort_values(by='매장개수', ascending=False).head(10)    
    st.dataframe(rank_region)  
    
    