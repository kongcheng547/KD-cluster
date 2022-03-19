### K-Means  (AgglomerativeClustering 为层次聚类使用,相应两行被替换)  向量距离聚类
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# from sklearn.cluster import AgglomerativeClustering

# 读取城市经纬度数据
X = []
f = open('2021/2021-01-01-test.csv')
for v in f:
    X.append([float(v.split(',')[0]), float(v.split(',')[1])])
# 转换成numpy array
X = np.array(X)
# print(X)
# 做归一化
#  以下两行参考http://blog.csdn.net/csj664103736/article/details/72828584
# a = X[:, :1] / 17075400.0 * 10000
# b = X[:, 1:] / 1392358258.0 * 10000
# print("a")
# print(a)
# print("b")
# print(b)
# X = np.concatenate((a, b), axis=1)
# 类簇的数量 - kmeans算法的参数
n_clusters = 5
# 进行聚类
cls = KMeans(n_clusters).fit(X)
# cls = AgglomerativeClustering(linkage='ward',n_clusters=n_clusters).fit(X)
# 聚类结果列表
cls.labels_
# 画图
markers = ['^', 'x', 'o', '*', '+']
colors = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']

for i in range(n_clusters):
    members = cls.labels_ == i
    plt.scatter(X[members, 0], X[members, 1], s=60, marker='o', c=colors[i], alpha=0.5)
# print(members)
# print(X[members,1])
plt.title('')
plt.show()
