# 2.2.2	变量和赋值
print('a' * 4)     #输出为：aaaa
s = "abc"
print('a' in s)     #输出为：True
print('d' in s)     #输出为：False
# （2）字符串处理常用函数
s = 'transport BIG DATA'      
len(s)          		#返回字符串长度，结果为18    
s.lower()       		#将字符串转为小写字母，结果为'transport big data'    
s.upper()       		#将字符串转为大写字母，结果为'TRANSPORT BIG DATA'    
s.capitalize()  		#首字母大写，结果为'Transport big data'    
s.strip('t')    		#去除字符串开头或结尾的指定字符，无参数时则去除空格、回车。结果为'ransport BIG DATA'    
s.split(' ')  			#以空格将字符串分割为列表，结果为['transport', 'BIG', 'DATA']    
s.count('a')    		#统计字符串中'a'出现的次数，结果为1    
s.find('a')     		#找到指定字符第一次出现的位置，结果为2    
'+'.join(['1', '2', '3'])  	#用指定字符将列表中字符连接，结果为'1+2+3'