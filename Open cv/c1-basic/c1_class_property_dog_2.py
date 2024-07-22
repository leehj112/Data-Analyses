# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:50:41 2024

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
# 데코레이터 : @property, @name.setter
# @property : 속성으로 선언
# @name.setter : name은 @property에 선언되어 있는 메소드 이름

#%%
class Dog:
    def __init__(self): # 생성자
        self.__ownernames = "default name"
        
    @property    
    def name(self): # getter
        return self.__ownernames
    
    @name.setter
    def name(self, name): # setter
        self.__ownernames = name
        
    
#%%

myDog = Dog()
print(myDog.name)

# 값을 할당하면 setter가 호출
myDog.name = "Marry" # Dog.name() setter 호출

# 값을 참조하면 getter가 호출
print(myDog.name)    # Dog.name() getter 호출

#%%

# TypeError: 'str' object is not callable
# myDog.name('길동')
# print(myDog.name())    
