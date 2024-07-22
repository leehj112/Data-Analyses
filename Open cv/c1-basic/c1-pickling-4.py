# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:48:10 2024

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



# 바이너리(이진) 파일
file = open("./pickling.txt", "rt")

r1 = file.readlines()
print(r1)
print(type(r1))

file.close()