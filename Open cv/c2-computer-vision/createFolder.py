# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:13:44 2024

@author: leehj
"""

import os

# 폴더 생성하기
def createFolder(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print ('Error: createFolder')