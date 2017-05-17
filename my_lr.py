# -*- coding: utf-8 -*-
'''
朴素贝叶斯分类器：
优点：
    1.计算代价不高,易于理解和实现
缺点：
    1.容易欠拟合,分类精度可能不高
适用数据类型:
    数值型和标称型
程序计算逻辑：
    1.
'''
import random

import numpy
from sklearn import preprocessing


def sigmoid(inx):
    return 1.0/(1+numpy.exp(-inx))


def sgd(data_feature_matric,class_labels,num_iter=150):
    item_count,feature_count = data_feature_matric.shape
    weights = numpy.ones(feature_count)
    for j in range(num_iter):
        data_index = range(item_count)
        for i in range(item_count):
            alpha = 4/(1.0+j+i)*0.01 + 0.0001
            alpha = (j+i)%10*0.00001 + 0.0001
            rand_index = random.randint(0,len(data_index)-1)
            h = sigmoid(numpy.sum(data_feature_matric[rand_index]*weights))
            error = class_labels[rand_index] - h
            weights = weights + alpha * error * data_feature_matric[rand_index]
            del data_index[rand_index]
    return weights

def gradAscent(dataMatIn, classLabels):
    dataMatrix = numpy.mat(dataMatIn)             #convert to NumPy matrix
    labelMat = numpy.mat(classLabels).transpose() #convert to NumPy matrix
    m,n = numpy.shape(dataMatrix)
    alpha = 0.0001
    maxCycles = 1000
    weights = numpy.ones((n,1))
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #matrix mult
        error = (labelMat - h)              #vector subtraction
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
    return weights

def classify_vector(inx,weights):
    prob = sigmoid(numpy.sum(inx*weights))
    return 1 if prob > 0.5 else 0

def run():
    train_feature_list = list()
    train_label_list = list()
    with open('bookdemo/Ch05/horseColicTraining.txt') as f1:
        for line in f1:
            curr_line = line.strip().split('\t')
            train_feature_list.append([float(i) for i in curr_line[:21]])
            train_label_list.append(float(curr_line[21]))
    scaler = preprocessing.MinMaxScaler().fit(numpy.array(train_feature_list))
    train_feature_list = scaler.transform(numpy.array(train_feature_list))
    train_weights = sgd(numpy.array(train_feature_list),train_label_list,1000)
    # train_weights = gradAscent(numpy.array(train_feature_list),train_label_list)
    # train_weights = stocGradAscent1(numpy.array(train_feature_list),train_label_list,1000)
    error_count = 0
    total_test_count = 0
    with open('bookdemo/Ch05/horseColicTest.txt') as f2:
        for line in f2:
            total_test_count += 1.0
            curr_line = line.strip().split('\t')
            t = scaler.transform(numpy.array([float(i) for i in curr_line[:21]]).reshape(1, -1))
            if classify_vector(t,train_weights) != int(float(curr_line[21])):
                error_count += 1
    error_rate = error_count/total_test_count
    print 'error count:%s,total count:%s,error rate:%s' % (error_count,total_test_count,error_rate)

if __name__=="__main__":
    run()
