# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 23:53:05 2024

@author: dlgur
"""

# 데이터베이스 연결 및 SQL 사용법
# ### 데이터베이스 연결 방법

#%%

# 데이터베이스: MySQL

#%%

# MySQL DB 연동용 패키지 설치
# get_ipython().system('pip install mysqlclient')
# pip install mysqlclient

#%%

# 패키지 불러오기
import MySQLdb

#%%

# db 연동 객체 만들기
# MySQL 연결하기 --- (※2)
conn = MySQLdb.connect(
    user='root',
    passwd='solsql',
    host='localhost',
    db='hellodb')

#%%

# 연결된 db에 sql 처리를 위한 cursor() 메서드 호출
cur = conn.cursor()

#%%

# 테이블 생성하기 --- (※4)
cur.execute('DROP TABLE IF EXISTS iris')
cur.execute('''
    CREATE TABLE iris (
        species VARCHAR(20),
        sl FLOAT(5,3),
        sw FLOAT(5,3),
        pl FLOAT(5,3),
        pw FLOAT(5,3)
    )
    ''')


#%%

# iris 데이터셋(데이터프레임) 불러오기
import seaborn as sns
iris = sns.load_dataset('iris')
iris.head()

#%%
import pandas as pd

# df = iris.rename(columns={'sepal_length' : 'sl', 'sepal_width': 'sw', 'petal_length' : 'pl', 'petal_width' : 'pw'})
iris.columns= ['sl', 'sw', 'pl', 'pw', 'species']

#%%

"""
for n in range(len(iris)):
    m = iris.iloc[n]
    sql = f"insert into iris (sl, sw, pl, pw, species) values ({m.iloc[0]}, {m.iloc[1]}, {m.iloc[2]}, {m.iloc[3]}, '{m.iloc[4]}')"
    cur.execute(sql)

conn.commit()
"""

#%%

for n in range(len(iris)):
    data = tuple(iris.iloc[n])
    sql = "insert into iris (sl, sw, pl, pw, species) values (%s, %s, %s, %s, %s)"
    print(data)
    cur.execute(sql, data)

conn.commit()

#%%

# ### 1-2. 데이터베이스 테이블의 데이터 조회(Select)


#%%

# DB iris 테이블에 전체 데이터 조회하기(select)
qry_s = "select * from iris"


#%%

# 쿼리 실행
cur.execute(qry_s)


#%%

# 쿼리 실행 결과 데이터 전체 조회해서 row에 저장
row = cur.fetchall()


#%%

row[0:5]


#%%


# row 데이터 행수 확인
len(row)


#%%

# row 데이터셋을 데이터프레임으로 변환
db_iris = pd.DataFrame(data = row, columns = ('sl', 'sw', 'pl', 'pw', 'species'))


#%%

# db에서 조회해 저장한 결과 확인
db_iris.head()


#%%

# ### 1-3. 데이터베이스 테이블의 데이터 입력(Insert)


#%%

# DB iris 테이블에 test 데이터 삽입하기(insert)
qry_i = "insert into iris (sl, sw, pl, pw, species) values (1, 2, 3, 4, 'test')"

#%%


# 쿼리 실행
cur.execute(qry_i)

#%%

# 쿼리 결과 확정
conn.commit()


#%%

# 데이터베이스 테이블의 데이터 삭제(Delete)

# DB iris 테이블에 test 데이터 삭제하기(delete)
qry_d = "delete from iris where species = 'test'"


#%%

# 쿼리 실행
cur.execute(qry_d)


#%%

# 쿼리 결과 확정
conn.commit()


#%%

# cursor 및 db 접속 종료
cur.close()
conn.close()

#%%

# THE END