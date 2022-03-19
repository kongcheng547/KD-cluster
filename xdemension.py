#!usr/bin/env python
# encoding:utf-8
from __future__ import division

import numpy as np

def simple_test():
    data_list = [[1, 2, 3], [1, 2, 1], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [6, 7, 9], [0, 4, 7], [4, 6, 0],
                 [2, 9, 1], [5, 8, 7], [9, 7, 8], [3, 7, 9]]
    # data_list.toarray()
    data_list = np.array(data_list)
    print('X[:,0]结果输出为：')
    print(data_list[:, 0])
    print('X[:,1]结果输出为：')
    print(data_list[:, 1])
    print('X[:,m:]结果输出为：(m=0,n=2)')
    print(data_list[:, 0:])
    data_list = [
        [[1, 2], [1, 0], [3, 4], [7, 9], [4, 0]],
        [[1, 4], [1, 5], [3, 6], [8, 9], [5, 0]],
        [[8, 2], [1, 8], [3, 5], [7, 3], [4, 6]],
        [[1, 1], [1, 2], [3, 5], [7, 6], [7, 8]],
        [[9, 2], [1, 3], [3, 5], [7, 67], [4, 4]],
        [[8, 2], [1, 9], [3, 43], [7, 3], [43, 0]],
        [[1, 22], [1, 2], [3, 42], [7, 29], [4, 20]],
        [[1, 5], [1, 20], [3, 24], [17, 9], [4, 10]],
        [[11, 2], [1, 110], [3, 14], [7, 4], [4, 2]]
    ]
    data_list = np.array(data_list)
    print('X[:,:,0]结果输出为：')
    print(data_list[:, :, 0])
    print('X[:,:,1]结果输出为：')
    print(data_list[:, :, 1])
    print('X[:,:,m:n]结果输出为：')
    print(data_list[:, :, 0:1])


if __name__ == '__main__':
    simple_test()