# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:47:35 2024

@author: leehj
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:37:45 2023

@author: Solero
"""

# P81
# 피클링(Pickling), 언피클링(Unpickling)
# 피클링: 파이썬 객체를 파일에 저장
# 언피클링: 파일에 저장된 파이썬 객체를 읽음

import pickle

group_name = "BlackPink"
member = 13
company = "YSIT미래교육원"
songs = { '마지막처럼': 2023, 'Kill this love':2019}

# 바이너리(이진) 파일
file = open("./pickling.dat", "wb")

pickle.dump(group_name, file)
pickle.dump(member, file)
pickle.dump(company, file)
pickle.dump(songs, file)

file.close()