# 2.8.3	Numpy简介
import numpy as np  
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) 	#定义一个ndarray对象
                                      #  [[1, 2, 3],
                                      #  [4, 5, 6],
                                      #  [7, 8, 9]]
a.ndim          					#ndarray的维数(阶)，输出为2  
a.shape         					#ndarray的形状，输出为(3, 3)，对应（行数，列数）
a.size          						#ndarray的元素个数，输出为9  
# 深复制浅复制
b = a                   				# 引用  
print(id(a)== id(b))  					# id()能够输出对象的内存地址，输出为True
c = a.view()            				# 浅复制
print(id(a)==id(c))					# 输出为False
d = a.copy()            				# 深复制  
# 多维数组的下标存取
a[0, 1]             					#输出2  (行、列的索引都是由0开始)
a[[0, 2], [1, 2]]   					#输出为array([2, 9])，等价于np.array([a[0, 1], a[2, 2]])  
a[:2, 1:3]          					#第0至1行，第1至2列array([[2, 3], [5, 6]])
a[2, 1:]            					#第2行，第1至最后1列  
a[:-1]              					#第0至倒数第2行  
a[1, :]             					#第1行，shape为(3,)，一维，array([4, 5, 6])
a[1:2, :]           					#第1行，shape为(1,3), 二维，array([[4, 5, 6]])
a[:,1]                                  #第1列，shape为(3, ), 一维，array([2, 5, 8])
a[:,1:2]                                #第1列，shape为(3, 1), 二维，array([[2], [5], [8]])
# Numpy中还能直接用判断条件（布尔矩阵）取出符合某些条件的元素：
a>2              					#输出为array([[False, False, True],  
                        			#		         [True, True, True],  
                        			#		         [True, True, True]])  
a[a>2]           					#输出数组中大于2的值，为一个一维数组 
a[(a>2) & (a<6)]                        #使用逻辑运算符连接多个条件，注意不能使用and,or之类的关键词进行连接，否则会报错
# 数组运算
#逐元素运算    
a + b    
a * b    
a/b     
a ** 2    
np.sin(a)    
# 矩阵运算    
a.dot(b)                   			#a与b矩阵相乘    
a = np.mat(a)              			#转换为矩阵对象    
a.I                        			#逆矩阵    
a.T                        			#转置    
a.trace()                  			#迹    
np.linalg.det(a)           				#矩阵a的行列式    
np.linalg.norm(a,ord=None) 				#矩阵a的范数    
np.linalg.eig(a)           				#矩阵a的特征值和特征向量    
np.linalg.cond(a,p=None)   				#矩阵a的条件数    
随机数生成
np.random.norm(0, 1, 100)  #生成均值为0，方差为1（不是标准差），长度为100的正态分布样本
np.random.poisson(5, 100)  #生成均值为5，长度为100的泊松分布样本
np.random.negative_binomial(1, 0.1, 100)  #生成n=1, p=0.1，长度为100的负二项分布样本