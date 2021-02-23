"""
11.3节
"""

a = np.mat([1, 3, 4])

def tanhFunc(z):
    
    return (1 - np.exp(-2 * z)) / (1 + np.exp(-2 * z))

def tanhDev(z):
    
    return 1 - np.square(tanFunc(z))

def reluFunc(z):
    
    tmp = z.copy()
    tmp[tmp<0] = 0
    
    return tmp

def reluDev(z):
    
    tmp = z.copy()
    tmp[tmp<0] = 0
    tmp[tmp>=0] = 1
    
    return tmp

def sigmoidFunc(z):   # sigmoid激活函数
    
    return 1 / (1 + np.exp(-z))

def sigmoidDev(z):
    
    return np.multiply(sigmoidFunc(z), (1 - sigmoidFunc(z)))

def softmaxFunc(z):   # softmax激活函数
    
    return np.exp(z) / sum(np.exp(z))

def calOneLayer(a, w, b, actFunc): # 计算一层的z值与a值
    
#     z = w.T * a + b         # 公式12.7
    z = w * a + b         # 公式12.7
    
    a_new = actFunc(z)      # 公式12.8
    
    return z, a_new

def calDeltaOutput(a, y): # 计算输出层误差项
    
    return a - y.T

def calDeltaHidden(delta, w, z, actDevFunc):   # 计算隐藏层误差项
    
    actDev =  actDevFunc(z)  # 激活函数的导数
    
    return  np.multiply(w.T * delta, actDev)  # 公式12.20

def oneHot(labels):      # 将标签转换为向量形式
    
    return np.eye(3)[y_test][0]

def calDevHidden(delta, a, alpha): # 计算隐藏层参数梯度
    
    return alpha * delta * a.T    # 公式12.19，计算偏置梯度时传入a=1

"""初始化参数"""

# def init(shape_weights, shape_bias):
#     np.random.seed(42)
#     weights = [np.mat(random(x.shape)) for x in clf.coefs_]

#     bias = [np.mat(random(x.shape)).T for x in clf.intercepts_ ]
    
#     return weights, bias

def init(shape_weights, shape_bias):  
    np.random.seed(42)  
    weights = [np.mat(random(x)) for x in shape_weights]  
  
    bias = [np.mat(random(x)) for x in shape_bias]  
      
    return weights, bias


"""前向"""

def forward(weights, bias):
    aList = []  # 保存a值
    zList = []  # 保存z值
    
    # 第2层,使用sigmoid激活函数
    z2, a2 = calOneLayer(x_test, weights[0], bias[0], sigmoidFunc)
    aList.append(a2)
    zList.append(z2)
    
    # 第3层，使用sigmoid激活函数
    z3, a3 = calOneLayer(a2, weights[1], bias[1], sigmoidFunc)
    aList.append(a3)
    zList.append(z3)
    
    # 第四层，输出层，使用softmax激活函数
    z4, a4 = calOneLayer(a3, weights[2], bias[2], softmaxFunc)
    aList.append(a4)
    zList.append(z4)
    
    return aList, zList

"""反向"""

def backward(aList, zList, alpha=0.02):
    
    dList = []
    
    delta4 = calDeltaOutput(aList[2], oneHot(y_test))      # 第4层，输出层误差项
    delta3 = calDeltaHidden(delta4, weights[2], zList[1], sigmoidDev)  # 第3层误差项　
    delta2 = calDeltaHidden(delta3, weights[1], zList[0], sigmoidDev)  # 第2层误差项

#     dev3 = calDevHidden(delta4, aList[1].T, alpha)  # 第3层权重梯度*学习率
    dev3 = calDevHidden(delta4, aList[1], alpha)  # 第3层权重梯度*学习率
#     dev3_bias = calDevHidden(delta4, 1, alpha)      # 第3层偏置梯度*学习率
    dev3_bias = calDevHidden(delta4, np.mat([1]), alpha)      # 第3层偏置梯度*学习率
    
#     dev2 = calDevHidden(delta3, aList[0].T, alpha)  # 第2层权重梯度*学习率
    dev2 = calDevHidden(delta3, aList[0], alpha)  # 第2层权重梯度*学习率    
#     dev2_bias = calDevHidden(delta3, 1, alpha)      # 第2层偏置梯度*学习率
    dev2_bias = calDevHidden(delta3, np.mat([1]), alpha)      # 第2层偏置梯度*学习率

#     dev1 = calDevHidden(delta2, x_test.T, alpha)    # 第1层权重梯度*学习率
    dev1 = calDevHidden(delta2, x_test, alpha)    # 第1层权重梯度*学习率
#     dev1_bias = calDevHidden(delta2, 1, alpha)      # 第1层偏置梯度*学习率
    dev1_bias = calDevHidden(delta2, np.mat([1]), alpha)      # 第1层偏置梯度*学习率
    
    return [dev1, dev2, dev3], [dev1_bias, dev2_bias, dev3_bias]

"""更新参数"""

def update(weights, bias, devList, devBiasList):
    [dev1, dev2, dev3], [dev1_bias, dev2_bias, dev3_bias] = devList, devBiasList
    
#     weights[0] = weights[0] - dev1.T  # 更新第1层权重
    weights[0] = weights[0] - dev1  # 更新第1层权重
    bias[0] = bias[0] - dev1_bias     # 更新第1层偏置

#     weights[1] = weights[1] - dev2.T   # 更新第2层权重
    weights[1] = weights[1] - dev2   # 更新第2层权重
    bias[1] = bias[1] - dev2_bias      # 更新第2层偏置

#     weights[2] = weights[2] - dev3.T   # 更新第3层权重
    weights[2] = weights[2] - dev3   # 更新第3层权重
    bias[2] = bias[2] - dev3_bias      # 更新第3层偏置
    
    return weights, bias


x_test = np.mat([[3.99475598], [0.17708362], [2.], [3.5008975 ], [3.]])  
y_test = np.mat([1])  
shape_weights = [(4, 5), (3, 4), (3, 3)]  # 各层权重矩阵大小  
shape_bias = [(4, 1), (3, 1), (3, 1)]   # 各层偏差矩阵大小  
weights, bias = init(shape_weights, shape_bias) #初始化  
for i in range(50)[:]:  
    aList, zList = forward(weights, bias)  
    devList, devBiasList = backward(aList, zList, 0.01) # 反向传播  
    weights, bias = update(weights, bias, devList, devBiasList)  # 更新参数  
