# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:37:47 2024

@author: leehj
"""


# 폴더 생성하기
def createFolder(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print ('Error: createFolder')