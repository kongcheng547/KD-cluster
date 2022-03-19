#引入数据集,sklearn包含众多数据集
from sklearn import datasets
#将数据分为测试集和训练集
from sklearn.model_selection import train_test_split
#利用邻近点方式训练数据
from sklearn.neighbors import KNeighborsClassifier

###引入数据###
#引入iris鸢尾花数据,iris数据包含4个特征变量
iris=datasets.load_iris()
iris_X=iris.data#特征变量
iris_y=iris.target#目标值
#利用train_test_split进行将训练集和测试集进行分开，test_size占30%
X_train,X_test,y_train,y_test=train_test_split(iris_X,iris_y,test_size=0.3)
#我们看到训练数据的特征值分为3类
print(y_train)
'''
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
 '''

###训练数据###
#引入训练方法
knn=KNeighborsClassifier()
#进行填充测试数据进行训练
knn.fit(X_train,y_train)

###预测数据###
#预测特征值
print(knn.predict(X_test))
'''
[1 1 1 0 2 2 1 1 1 0 0 0 2 2 0 1 2 2 0 1 0 0 0 0 0 0 2 1 0 0 0 1 0 2 0 2 0
 1 2 1 0 0 1 0 2]
'''
#真实特征值
print(y_test)
'''
[1 1 1 0 1 2 1 1 1 0 0 0 2 2 0 1 2 2 0 1 0 0 0 0 0 0 2 1 0 0 0 1 0 2 0 2 0
 1 2 1 0 0 1 0 2]
'''
