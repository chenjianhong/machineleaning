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

def sigmoid(inx):
    return 1.0/(1+numpy.exp(-inx))

def sgd(data_feature_matric,class_labels,num_iter=150):
    item_count,feature_count = data_feature_matric.shape
    weights = numpy.ones(feature_count)
    for j in range(num_iter):
        data_index = range(item_count)
        for i in range(item_count):
            alpha = 4/(1.0+j+i) + 0.0001
            rand_index = random.randint(0,len(data_index)-1)
            h = sigmoid(sum(data_feature_matric[rand_index]*weights))
            error = class_labels[rand_index] - h
            weights = weights + alpha * error * data_feature_matric[rand_index]
            del data_index[rand_index]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = dataMatrix.shape
    weights = numpy.ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def classify_vector(inx,weights):
    prob = sigmoid(sum(inx*weights))
    return 1 if prob > 0.5 else 0

def run():
    train_feature_list = list()
    train_label_list = list()
    with open('bookdemo/Ch05/horseColicTraining.txt') as f1:
        for line in f1:
            curr_line = line.strip().split('\t')
            train_feature_list.append([float(i) for i in curr_line[:21]])
            train_label_list.append(float(curr_line[21]))
    # train_weights = sgd(numpy.array(train_feature_list),train_label_list,1000)
    train_weights = stocGradAscent1(numpy.array(train_feature_list),train_label_list,1000)
    error_count = 0
    total_test_count = 0
    with open('bookdemo/Ch05/horseColicTest.txt') as f2:
        for line in f2:
            total_test_count += 1.0
            curr_line = line.strip().split('\t')
            if classify_vector(numpy.array([float(i) for i in curr_line[:21]]),train_weights) != int(float(curr_line[21])):
                error_count += 1
    error_rate = error_count/total_test_count
    print 'error count:%s,total count:%s,error rate:%s' % (error_count,total_test_count,error_rate)

def colicTest():
    frTrain = open('bookdemo/Ch05/horseColicTraining.txt'); frTest = open('bookdemo/Ch05/horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = sgd(numpy.array(trainingSet), trainingLabels, 1000)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classify_vector(numpy.array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate

if __name__=="__main__":
    colicTest()
