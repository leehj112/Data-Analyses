# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:38:31 2024

@author: leehj
"""


# 누락 데이터확인 
import seaborn as sns

df = sns.load_dataset('titanic') 

#%%

nan_deck = df['deck'].value_counts(dropna=False)
print(nan_deck) 

"""
deck
NaN    688
C       59
B       47
D       33
E       32
A       15
F       13
G        4
Name: count, dtype: int64
"""
#%% 
print(df.head().isnull()) 
# 누락 데이터를 찾는 직접적인 방법: isnull(), notnull 
"""
   survived  pclass    sex    age  ...   deck  embark_town  alive  alone
0     False   False  False  False  ...   True        False  False  False
1     False   False  False  False  ...  False        False  False  False
2     False   False  False  False  ...   True        False  False  False
3     False   False  False  False  ...  False        False  False  False
4     False   False  False  False  ...   True        False  False  False

[5 rows x 15 columns]
"""
#%%
print(df.head().notnull()) 
"""
survived  pclass   sex   age  ...   deck  embark_town  alive  alone
0      True    True  True  True  ...  False         True   True   True
1      True    True  True  True  ...   True         True   True   True
2      True    True  True  True  ...  False         True   True   True
3      True    True  True  True  ...   True         True   True   True
4      True    True  True  True  ...  False         True   True   True

[5 rows x 15 columns]
"""
#%%
print(df.head().isnull().sum(axis=0)) 
"""
survived       0
pclass         0
sex            0
age            0
sibsp          0
parch          0
fare           0
embarked       0
class          0
who            0
adult_male     0
deck           3
embark_town    0
alive          0
alone          0
dtype: int64
"""
#%% 
# 누락 데이터 제거 
#  --> titanic에 누락데이터가 몇 개씩 포함되어 있는지 check 
import seaborn as sns

df = sns.load_dataset('titanic')

missing_df = df.isnull()
for col in missing_df.columns:
    missing_count = missing_df[col].value_counts()
    
    try:
        print(col,':', missing_count[True])
    except:
        print(col,': ', 0) 
    
"""
survived :  0
pclass :  0
sex :  0
age : 177
sibsp :  0
parch :  0
fare :  0
embarked : 2
class :  0
who :  0
adult_male :  0
deck : 688
embark_town : 2
alive :  0
alone :  0
"""
# 전체 891명 중 688명에 데크 데이터가 누락 ==> NaN값을 500개 이상 갖는 모든 열을 삭제 
#%%
# 누락 데이터 제거 
# NaN값 500개 이상인 열을 모두 삭제 -> deck 열(891개 중 688개의 NaN값) 
df_thresh = df.dropna(axis=1, thresh=500)
print(df_thresh.columns) 
"""
Index(['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',
       'embarked', 'class', 'who', 'adult_male', 'embark_town', 'alive',
       'alone'],
      dtype='object')
"""
#%% 
# 891개에 행중 나이 데이터가 누락된 177개행 삭제 나머지 714개 --> df_age에 저장 
df_age = df.dropna(subset=['age'], how='any', axis=0)
print(len(df_age)) 

# 714 
#%% 
import seaborn as sns
df = sns.load_dataset('titanic')

# age열의 첫 10개 데이터 출력(5행에 NaN 값) 
print(df['age'].head(10)) 
print('\n') 

# age열의 NaN값을 다른 나이 데이터의 평균으로 변경
mean_age = df['age'].mean(axis=0)
df['age'].fillna(mean_age, inplace=True)

# age 열의 첫 10개 데이터 출력(5행에 NaN 값이 평균으로 대체) 
print(df['age'].head(10)) 
"""
0    22.0
1    38.0
2    26.0
3    35.0
4    35.0
5     NaN
6    54.0
7     2.0
8    27.0
9    14.0
Name: age, dtype: float64


0    22.000000
1    38.000000
2    26.000000
3    35.000000
4    35.000000
5    29.699118
6    54.000000
7     2.000000
8    27.000000
9    14.000000
"""
#%% 
# 시계열 데이터 
# 자료형 시간 데이터 -> 판다스 시계열 객체인 Timestamp로 변환하는 함수 제공 

import pandas as pd

df = pd.read_csv('./data/part5/stock-data.csv')


# 데이터 내용 및 자료형 확인 
print(df.head())
print('\n')
print(df.info()) 
"""
        Date  Close  Start   High    Low  Volume
0  2018-07-02  10100  10850  10900  10000  137977
1  2018-06-29  10700  10550  10900   9990  170253
2  2018-06-28  10400  10900  10950  10150  155769
3  2018-06-27  10900  10800  11050  10500  133548
4  2018-06-26  10800  10900  11000  10700   63039


<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20 entries, 0 to 19
Data columns (total 6 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   Date    20 non-null     object
 1   Close   20 non-null     int64 
 2   Start   20 non-null     int64 
 3   High    20 non-null     int64 
 4   Low     20 non-null     int64 
 5   Volume  20 non-null     int64 
dtypes: int64(5), object(1)
memory usage: 1.1+ KB
None
"""
#%% 
df['new_Date'] = pd.to_datetime(df['Date']) 

print(df.head()) 
print('\n') 
print(df.info()) 
print('\n') 
print(type(df['new_Date'][0])) 



"""
         Date  Close  Start   High    Low  Volume   new_Date
0  2018-07-02  10100  10850  10900  10000  137977 2018-07-02
1  2018-06-29  10700  10550  10900   9990  170253 2018-06-29
2  2018-06-28  10400  10900  10950  10150  155769 2018-06-28
3  2018-06-27  10900  10800  11050  10500  133548 2018-06-27
4  2018-06-26  10800  10900  11000  10700   63039 2018-06-26


<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20 entries, 0 to 19
Data columns (total 7 columns):
 #   Column    Non-Null Count  Dtype         
---  ------    --------------  -----         
 0   Date      20 non-null     object        
 1   Close     20 non-null     int64         
 2   Start     20 non-null     int64         
 3   High      20 non-null     int64         
 4   Low       20 non-null     int64         
 5   Volume    20 non-null     int64         
 6   new_Date  20 non-null     datetime64[ns]
dtypes: datetime64[ns](1), int64(5), object(1)
memory usage: 1.2+ KB
None

"""
#%% 
# 시계열 데이터 시간 순서에 맞춰 인데싱, 슬라이싱 편리 
df.set_index('new_Date', inplace=True)
df.drop('Date', axis=1, inplace=True)    # date drop 

print(df.head())
print('\n')
print(df.info()) 
"""
            Close  Start   High    Low  Volume
new_Date                                      
2018-07-02  10100  10850  10900  10000  137977
2018-06-29  10700  10550  10900   9990  170253
2018-06-28  10400  10900  10950  10150  155769
2018-06-27  10900  10800  11050  10500  133548
2018-06-26  10800  10900  11000  10700   63039


<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 20 entries, 2018-07-02 to 2018-06-01
Data columns (total 5 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   Close   20 non-null     int64
 1   Start   20 non-null     int64
 2   High    20 non-null     int64
 3   Low     20 non-null     int64
 4   Volume  20 non-null     int64
dtypes: int64(5)
memory usage: 960.0 bytes
None
"""
#%% 
# TimeStamp를 Peiod로 변환 
import pandas as pd 

dates = ['2019-01-01', '2020-03-01', '2021-06-01']

ts_dates = pd.to_datetime(dates)
print(ts_dates)
print('\n')

# timestamp ==> period로 변환 
pr_day = ts_dates.to_period(freq='D')
print(pr_day)

pr_month = ts_dates.to_period(freq='M')
print(pr_month)

pr_year = ts_dates.to_period(freq='A')
print(pr_year) 
"""
DatetimeIndex(['2019-01-01', '2020-03-01', '2021-06-01'], dtype='datetime64[ns]', freq=None)


PeriodIndex(['2019-01-01', '2020-03-01', '2021-06-01'], dtype='period[D]')
PeriodIndex(['2019-01', '2020-03', '2021-06'], dtype='period[M]')
PeriodIndex(['2019', '2020', '2021'], dtype='period[Y-DEC]')
"""
#%%
# Timestamp 배열 
import pandas as pd


# Timesetamp의 배열 - 월,간격,월의 시작일 기준 
ts_ms = pd.date_range(start = '2019-01-01',   # 날짜 범위 시작 
                      end=None,               # 날짜 범위 끝 
                      periods=6,              # 생성할 Timestamp 개수 
                      freq='MS',              # 시간간격 
                      tz='Asia/Seoul')        # 시간대(timezone) 

print(ts_ms) 
"""
DatetimeIndex(['2019-01-01 00:00:00+09:00', '2019-02-01 00:00:00+09:00',
               '2019-03-01 00:00:00+09:00', '2019-04-01 00:00:00+09:00',
               '2019-05-01 00:00:00+09:00', '2019-06-01 00:00:00+09:00'],
              dtype='datetime64[ns, Asia/Seoul]', freq='MS')
"""

#%%
# 월 간격, 월의 마지막 날 기준 
ts_me = pd.date_range('2018-01-01', periods=6,   
                      freq='M',                  # M은 월 표시         
                      tz='Asia/Seoul')

print(ts_me) 
print('\n')

#%% 

# 분기(3개월) 간격, 월의 마지막 날 기준 
ts_3m = pd.date_range('2019-01-01', periods=6,
                      freq='3M',
                      tz='Asia/Seoul') 
print(ts_3m) 

"""
DatetimeIndex(['2018-01-31 00:00:00+09:00', '2018-02-28 00:00:00+09:00',
               '2018-03-31 00:00:00+09:00', '2018-04-30 00:00:00+09:00',
               '2018-05-31 00:00:00+09:00', '2018-06-30 00:00:00+09:00'],
              dtype='datetime64[ns, Asia/Seoul]', freq='ME')


DatetimeIndex(['2019-01-31 00:00:00+09:00', '2019-04-30 00:00:00+09:00',
               '2019-07-31 00:00:00+09:00', '2019-10-31 00:00:00+09:00',
               '2020-01-31 00:00:00+09:00', '2020-04-30 00:00:00+09:00'],
              dtype='datetime64[ns, Asia/Seoul]', freq='3ME')

"""
#%% 
# Period 배열 
# 판다스 period_range()함수는 여러 개의 기간이 들어 있는 시계열 데이터를 만든다. 

import pandas as PD 

pr_m = pd.period_range(start='2019-01-01',
                       end=None,
                       periods=3,
                       freq='M')

print(pr_m) 
"""
PeriodIndex(['2019-01', '2019-02', '2019-03'], dtype='period[M]')
"""
#%% 
# h = 1시간 
pr_h = pd.period_range(start='2019-01-01',
                       end=None,
                       periods=3,
                       freq='H')

print(pr_h) 
print('\n')
"""
PeriodIndex(['2019-01-01 00:00', '2019-01-01 01:00', '2019-01-01[02:00']
"""
#%% 
# h = 2시간
pr_2h = pd.period_range(start='2019-01-01',
                        end=None,
                        periods=3,
                        freq='2H')

print(pr_2h) 
"""
PeriodIndex(['2019-01-01 00:00', '2019-01-01 02:00', '2019-01-01 04:00'], dtype='period[2h]')
"""
#%% 
# 날짜 데이터 분리 
import pandas as pd 

df = pd.read_csv('./data/part5/stock-data.csv') 

df['new_Date'] = pd.to_datetime(df['Date'])
print(df.head())
print('\n') 

"""
    Date  Close  Start   High    Low  Volume   new_Date
0  2018-07-02  10100  10850  10900  10000  137977 2018-07-02
1  2018-06-29  10700  10550  10900   9990  170253 2018-06-29
2  2018-06-28  10400  10900  10950  10150  155769 2018-06-28
3  2018-06-27  10900  10800  11050  10500  133548 2018-06-27
4  2018-06-26  10800  10900  11000  10700   63039 2018-06-26
"""

# dt 속성을 이용하여 new_Date 열의 연-월-일 정보를 년, 월, 일로 구분 
df['Year'] = df['new_Date'].dt.year
df['Month'] = df['new_Date'].dt.month
df['Day'] = df['new_Date'].dt.day

print(df.head()) 
"""
         Date  Close  Start   High    Low  Volume   new_Date  Year  Month  Day
0  2018-07-02  10100  10850  10900  10000  137977 2018-07-02  2018      7    2
1  2018-06-29  10700  10550  10900   9990  170253 2018-06-29  2018      6   29
2  2018-06-28  10400  10900  10950  10150  155769 2018-06-28  2018      6   28
3  2018-06-27  10900  10800  11050  10500  133548 2018-06-27  2018      6   27
4  2018-06-26  10800  10900  11000  10700   63039 2018-06-26  2018      6   26
"""

#%% 
# Timestamp를 period로 변환 --> 연-월-일 표기 변경 
df['Date_yr'] = df['new_Date'].dt.to_period(freq='A')
df['Date_m'] = df['new_Date'].dt.to_period(freq='M')
print(df.head()) 
"""
          Date  Close  Start   High    Low  ...  Year Month  Day  Date_yr   Date_m
0  2018-07-02  10100  10850  10900  10000  ...  2018     7    2     2018  2018-07
1  2018-06-29  10700  10550  10900   9990  ...  2018     6   29     2018  2018-06
2  2018-06-28  10400  10900  10950  10150  ...  2018     6   28     2018  2018-06
3  2018-06-27  10900  10800  11050  10500  ...  2018     6   27     2018  2018-06
4  2018-06-26  10800  10900  11000  10700  ...  2018     6   26     2018  2018-06
"""
#%% 
df.set_index('Date_m', inplace=True)
print(df.head()) 

"""
              Date  Close  Start   High  ...  Year  Month Day  Date_yr
Date_m                                    ...                          
2018-07  2018-07-02  10100  10850  10900  ...  2018      7   2     2018
2018-06  2018-06-29  10700  10550  10900  ...  2018      6  29     2018
2018-06  2018-06-28  10400  10900  10950  ...  2018      6  28     2018
2018-06  2018-06-27  10900  10800  11050  ...  2018      6  27     2018
2018-06  2018-06-26  10800  10900  11000  ...  2018      6  26     2018

"""
#%% 
# 날짜 인덱스 활용 
import pandas as pd

df = pd.read_csv('./data/part5/stock-data.csv') 

df['new_Date'] = pd.to_datetime(df['Date'])
df.set_index('new_Date', inplace=True)          # 행 인덱스로 지정 

print(df.head())
print('\n')
print(df.index) 

"""
                 Date  Close  Start   High    Low  Volume
new_Date                                                  
2018-07-02  2018-07-02  10100  10850  10900  10000  137977
2018-06-29  2018-06-29  10700  10550  10900   9990  170253
2018-06-28  2018-06-28  10400  10900  10950  10150  155769
2018-06-27  2018-06-27  10900  10800  11050  10500  133548
2018-06-26  2018-06-26  10800  10900  11000  10700   63039


DatetimeIndex(['2018-07-02', '2018-06-29', '2018-06-28', '2018-06-27',
               '2018-06-26', '2018-06-25', '2018-06-22', '2018-06-21',
               '2018-06-20', '2018-06-19', '2018-06-18', '2018-06-15',
               '2018-06-14', '2018-06-12', '2018-06-11', '2018-06-08',
               '2018-06-07', '2018-06-05', '2018-06-04', '2018-06-01'],
              dtype='datetime64[ns]', name='new_Date', freq=None
"""
#%% 
# 시간 간격 계산, 최근 180~189일 사이의 값들 선택 
today = pd.to_datetime('2018-12-25')    # 기준일 생성 
df['time_delta'] = today - df.index     # 날짜 차이 계산 
df.set_index('time_delta', inplace=True)  # 행 인덱스로 지정 
df_180 = df['180 days': '189 days']
print(df_180) 

"""
    Date  Close  Start   High    Low  Volume
time_delta                                                
180 days    2018-06-28  10400  10900  10950  10150  155769
181 days    2018-06-27  10900  10800  11050  10500  133548
182 days    2018-06-26  10800  10900  11000  10700   63039
183 days    2018-06-25  11150  11400  11450  11000   55519
186 days    2018-06-22  11300  11250  11450  10750  134805
187 days    2018-06-21  11200  11350  11750  11200  133002
188 days    2018-06-20  11550  11200  11600  10900  308596
189 days    2018-06-19  11300  11850  11950  11300  180656
"""
#%% 



