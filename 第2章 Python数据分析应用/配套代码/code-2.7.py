# 2.7	匿名函数
def add(x):         #定义加1函数  
    return x + 1  
num = [1, 2, 3]  
  
list(map(add, num)) #返回[2, 3, 4]，由于map函数返回的是map对象，需要使用list函数转换为list 

num = [1, 2, 3]  
list(map(lambda x: x+1, num)) #返回[2, 3, 4]
