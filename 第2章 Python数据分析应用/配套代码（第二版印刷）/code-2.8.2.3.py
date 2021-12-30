# random模块
import random    
random.random()         					#生成0-1之间均匀分布的随机浮点数
random.normalvariate(0, 1) 					#生成1个符合均值为0，方差为1正态分布的随机数
[random.normalvariate(0, 1) for x in range(10)] 	#生成长度为10的正态分布序列
random.uniform(a, b)    					#生成[a, b]区间内的随机浮点数
random.randint(a, b)    					#生成[a, b]区间内的随机整数
random.choice(s)         					#从序列s中随机获取一个值
random.shuffle(s)        					#将序列s中元素打乱
random.sample(s, k)      					#从序列s中获取长度为k的片段