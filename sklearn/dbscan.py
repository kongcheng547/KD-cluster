### DBScan算法  密度聚类
# coding = utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

X = []
f = open('../2021/2021-01-01-test.csv')
for v in f:
    X.append([float(v.split(',')[0]), float(v.split(',')[1])])
# 转换成numpy array
X = np.array(X)
# 做归一化
#  以下两行参考http://blog.csdn.net/csj664103736/article/details/72828584
a = X[:, :1]
b = X[:, 1:]
# print("a")
# print(a)
# print("b")
# print(b)
X = np.concatenate((a, b), axis=1)
#  numpy.concatenate((a1,a2,...), axis=0)函数。能够一次完成多个数组的拼接,axis默认为0，表示直接拼接；axis=1表示对应列拼接
# 进行数据训练(两参数：扫描半径，最小包含点数)
cls = DBSCAN(eps=1, min_samples=15).fit(X)
# 类簇的数量
# len() 方法返回对象（字符、列表、元组等）长度或项目个数。
# set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
n_clusters = len(set(cls.labels_))
# print(n_clusters) 结果为 5
# X中每项所属分类的一个列表
cls.labels_
# 画图
markers = ['^', 'x', 'o', '*', '+']
colors = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']
for i in range(n_clusters):
    my_members = cls.labels_ = i
    plt.scatter(X[my_members, 0], X[my_members, 1], s=60, marker=markers[i], c=cls, alpha=0.5)
plt.title('dbscan')
plt.show()
