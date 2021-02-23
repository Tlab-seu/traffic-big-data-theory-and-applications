# 第一步 ：数据生成
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
import numpy as np

X, y = datasets.make_regression(n_samples=100, n_features=1,
                                n_informative=1, noise=2, random_state=9)
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='b')
# 调整绘图的样式
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.grid(linestyle='-.')
ax = plt.gca()
ax.xaxis.set_label_coords(1.02, 0.04)
ax.yaxis.set_label_coords(-0.04, 1)
# 生成散点图
plt.show()
# 第二步：训练集、测试集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10)
# 第三步：标准化
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X = scaler.transform(X)
# 第四步：模型训练
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
# 模型预测
y_predict = lin_reg.predict(X_test)
# 模型R2系数
lin_r2_score = lin_reg.score(X_test, y_test)
print(lin_r2_score)  # 输出：0.8714
# 第五步：模型评价
lin_mse = mean_squared_error(y_test, y_predict)
print(lin_mse)  # 输出：2.4945
# 第六步：10折交叉验证
predicted = cross_val_predict(lin_reg, X, y, cv=10)
cv_lin_mse = mean_squared_error(y, predicted)
print(cv_lin_mse)  # 输出：3.5290
# 第七步：可视化
# 将标准化的X还原
X = scaler.inverse_transform(X)
X_min = min(X)[0]
X_max = max(X)[0]
X_line = np.linspace(X_min, X_max, 1000)
y_line = X_line * lin_reg.coef_ + lin_reg.intercept_
plt.figure(figsize=(8, 6))
ax = plt.gca()
ax.scatter(X, y, color='b')
ax.plot(X_line, y_line, 'r', lw=4)
ax.set_title('Linear Regression', fontsize=20)
ax.set_xlabel('x', fontsize=20)
ax.set_ylabel('y', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.xaxis.set_label_coords(1.02, 0.04)
ax.yaxis.set_label_coords(-0.04, 1)
plt.grid(linestyle='-.')
plt.show()
