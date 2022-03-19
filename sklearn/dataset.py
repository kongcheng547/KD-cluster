from sklearn import datasets  # 引入数据集

# 构造的各种参数可以根据自己需要调整
X, y = datasets.make_regression(n_samples=200, n_features=2, n_targets=2, noise=0)

###绘制构造的数据###
import matplotlib.pyplot as plt

plt.figure()
plt.scatter(X, y)
plt.show()