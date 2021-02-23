from tensorflow import keras
vocab_size = 10000          #词汇表的大小，可理解为一个参数，目前无需深究
#步骤一：模型定义
model = keras.Sequential() 
model.add(keras.layers.Embedding(vocab_size, 16)) 		#添加Embedding层
model.add(keras.layers.GlobalAveragePooling1D())  		#添加池化层
model.add(keras.layers.Dense(16, activation=tf.nn.relu)) 	#添加全连接层
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid)) #添加全连接层
model.summary() 									#输出模型结构
#步骤二：模型编译
model.compile(optimizer='adam',     					#使用adam方法进行参数优化
              loss='binary_crossentropy', 			#损失函数选用二元交叉熵
              metrics=['acc'])             			#评价指标采用准确率
x_val = trainData[:10000]              				#训练集中前10000个样本作为验证集
partial_x_train = trainData[10000:]
y_val = trainLabels[:10000]
partial_y_train = trainLabels[10000:]
#步骤三：模型训练
history = model.fit(partial_x_train,  					#输入自变量
                    partial_y_train,    				#输入因变量（标签）
                    epochs=40,           			#训练次数为40
                    batch_size=512,      			#分批训练，每批样本数为512
                    validation_data=(x_val, y_val),
                    verbose=1)            		#显示每次训练的进度
#步骤四：模型校验
results = model.evaluate(testData, testLabels)
print(results)