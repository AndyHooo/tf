# _*_ coding: utf-8 _*_
'''
Created on Jan 24, 2018

@author: hudaqiang
'''
class Person(object):
    #getter
    @property
    def age(self):
        return self._age
    
    #setter
    @age.setter
    def age(self,age):
        self._age = age
    
    @staticmethod
    def hello():
        print 'hello'
        
    @classmethod
    def say_hello(cls):
        print 'say hello'
Person.hello()
Person.say_hello()
person = Person()
person.age = 16
print person.age