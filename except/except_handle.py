# -*_ coding: utf-8 _*_s
'''
Created on Jan 24, 2018

@author: hudaqiang
'''
class MyException(Exception):
    def __init__(self,message):
        self.message = message
        
if __name__ == '__main__':
    try:
        1/0
    except Exception:
        raise MyException('除数不能为0')
    finally:
        print '大哥学过数学没有!'
        
    