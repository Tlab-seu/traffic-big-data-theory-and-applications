import time
start = time.time()          #记录代码开始运行时间，time.time()用于获取当前时间戳
a = 1566897866         	#类型1：时间戳数据，常见的时间输入格式
c = time.localtime(a)  		#将时间戳转换为时间格式
print(c)  
# 输出结果如下：time.struct_time(tm_year=2019, tm_mon=8, tm_mday=27, tm_hour=17, tm_min=24, tm_sec=26, tm_wday=1, tm_yday=239, tm_isdst=0)
a = "2017-6-11 17:51:30"  	#类型2：字符型时间数据
c=time.strptime(a,"%Y-%m-%d %H:%M:%S")
print(c)  
# 输出结果如下：  
# time.struct_time(tm_year=2017, tm_mon=6, tm_mday=11, tm_hour=17, tm_min=51, tm_sec=30, tm_wday=6, tm_yday=162, tm_isdst=-1)
c.tm_year  				#年
c.tm_mon   			#月
c.tm_mday  			#日
c.tm_hour  			#小时
c.tm_min   			#分钟
c.tm_sec   			#秒
c.tm_wday  			#星期几
c.tm_yday  			#到当年1月1日的天数
end = time.time()         #运行结束时间
timeSpent = end - start    #计算运行时间
print("Time spent: {0} s".format(timeSpent))