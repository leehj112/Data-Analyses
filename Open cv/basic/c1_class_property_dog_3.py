# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:51:11 2024

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
    myname = '호빵이'  # 클래스 변수(static variable)
    
    def __init__(self): # 생성자
        self.__myname = "default name"
        
  
    def name(self): # getter
        return self.__myname
    
    # name() getter는 덮어쓰기 되어 없어짐
    def name(self, name): # setter
        self.__myname = name
        print(self.__myname)

        
        
#%%
print(Dog.myname)

#%%

# Dog 클래스로 myDog 객체를 생성
myDog = Dog()

# AttributeError: 'Dog' object has no attribute '__myname'
print(myDog.__myname) # 접근할 수 없음

#%%

# 
namefunc = myDog.name
namefunc("진도개")
