# _*_ coding: utf-8 _*_
'''
Created on 2018年1月29日
测试python调用其他语言
@author: hudaqiang
'''
from outer_invoke.java_invoke import JavaInvocation
import jpype
if __name__ == '__main__':
    
    javaInvocation = JavaInvocation('commons-io-2.4.jar')
    javaInvocation.start_jvm()
    
    #java和javax两个库可以包名.类名调用
    salt_file = jpype.java.io.File('/Users/hudaqiang/Downloads/resource/salt')
    fileUtils = jpype.JPackage('org').apache.commons.io.FileUtils
    read_lines = fileUtils.readLines(salt_file)
    javaInvocation.jprint(read_lines)
    
    javaInvocation.shutdown_jvm()
