# 2.6	异常处理
try:    
    a = 1 / 0    
except ZeroDivisionError as e:     # 此处e为ZeroDivisionError的别名，可通过它获得更详细的信息
    a = 1
    print(e)                    # 输出为 division by zero 
finally:  
    print(a)