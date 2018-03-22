# _*_ coding: utf-8 _*_
'''
Created on 2018年1月28日
调用java代码
@author: hudaqiang
'''
import jpype 
import os.path

class JavaInvocation(object):
    """
    python调用java 
    """
    
    JVM_PATH = jpype.getDefaultJVMPath()
    JAVA_LIB = '/Users/hudaqiang/eclipse-workspace/tf/outer_libs/java/'
    
    def __init__(self,jar_name):
        self._jar_name = jar_name
    
    def start_jvm_default(self):
        jpype.startJVM(self.JVM_PATH)  
        
    def start_jvm(self):
        jarpath = os.path.join(os.path.abspath('.'), self.JAVA_LIB)  
        jpype.startJVM(self.JVM_PATH,"-ea", "-Djava.class.path=%s" % (jarpath + self._jar_name))  
        
    def shutdown_jvm(self):
        jpype.shutdownJVM() 
        
    @staticmethod
    def jprint(message):
        jprint = jpype.java.lang.System.out.println  
        jprint(message) 
