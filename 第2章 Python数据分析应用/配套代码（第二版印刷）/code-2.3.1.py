# 2.3.1	列表
#（1）列表的定义
station_names = ['a1', 'a2', 'a3', 'a4']      
len(station_names)      #len()为计算列表长度的函数，输出结果为4      
#（2）列表的取值和切片
station_names[1]        	#第1个元素，结果为'a2'  
station_names[1:3]      	#第1-2个元素，结果为['a2', 'a3']    
station_names[:3]       	#第0-2个元素，结果为['a1', 'a2', 'a3']    
station_names[2:]       	#第2至最后一个元素，结果为['a3', a4']    
station_names[-1]       	#最后一个元素，结果为'a4'    
station_names[:-2]      	#第0个至倒数第3个元素，结果为['a1', 'a2'] 
#（3）元素的增删
station_names.append('a4')   	#添加到末尾，结果为 ['a1', 'a2', 'a3', 'a4', 'a4']
station_names.insert(1, 'a4')   	#插入到指定位置，结果为['a1', 'a4', 'a2', 'a3', 'a4']
station_names.pop()      		#删除末尾元素，处理后列表为['a1', 'a2', 'a3']，返回值为末尾元素
station_names.pop(2)     		#删除指定索引处元素，处理后列表为['a1', 'a2', 'a4']，返回指定索引处元素  
#（4）列表的生成
a = [1, 2, 3]
b = [x + 2 for x in a] 			#b的返回值为[3, 4, 5]
c = {'a':1, 'b':3, 'c':7}
d = [(x, y+1) for x, y in c.items()] 	#返回[('b', 4), ('c', 8), ('a', 2)]