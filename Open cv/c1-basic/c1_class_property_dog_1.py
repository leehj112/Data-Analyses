# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:50:21 2024

@author: leehj
"""

# -*- coding: utf-8 -*-
"""
Version: 4.8.1.78

conda activate YSIT23
pip install opencv-python
pip install opencv-contrib-python

# P76 클래스
"""

# 프라퍼티(Property) 활용

#%%
class Dog:
    def __init__(self): # 생성자
        self.__ownernames = "default name"
        
    def get_name(self): # getter
        return self.__ownernames
    
    def set_name(self, name): # setter
        self.__ownernames = name
        
    
#%%

myDog = Dog()
print(myDog.get_name())
myDog.set_name("Marry")

print(myDog.get_name())
