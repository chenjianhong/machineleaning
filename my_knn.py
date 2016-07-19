# coding: utf-8
'''
k 近邻算法：
优点：
1.精度高、对异常值不敏感，无数据输入假定
缺点：
1.计算复杂度高、空间复杂度高
'''
import os
import zipfile
import numpy as np
import operator


def img2vector(file_hander):
    vect = np.zeros(1024)
    for i in range(0,32):
        line = file_hander.readline()
        for j in range(0,32):
            vect[32*i+j] = int(line[j])
    return vect


def classify(x,data_set,labels,k):
    data_set_size = data_set.shape[0]  # 获取行数
    diff_mat = np.tile(x,(data_set_size,1)) - data_set  # 将数组重复data_set_size维，每个数组里重复1次
    sq_diff_mat = diff_mat**2
    sq_distances = sq_diff_mat.sum(axis=1) # 行向量相加
    distances = sq_distances**0.5
    sorted_dis_indicies = distances.argsort()  # 将数组排序并返回索引数组
    class_count = dict()
    for i in range(k):
        vote_label = labels[sorted_dis_indicies[i]]
        class_count[vote_label] = class_count.get(vote_label,0) + 1
    sorted_class_count = sorted(class_count.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sorted_class_count[0][0]



def run():
    z = zipfile.ZipFile("./bookdemo/Ch02/digits.zip")
    class_label = list()
    train_vect = list()
    for f in z.namelist():
        if not f.endswith('/') and f.startswith('trainingDigits'):
            with z.open(f) as img_f:
                class_number = f.split('/')[-1].split('.')[0].split('_')[0]
                class_label.append(class_number)
                train_vect.append(img2vector(img_f))
    train_vect = np.asarray(train_vect)
    class_label = np.asarray(class_label)

    error_count = 0
    total_count = 0
    for f in z.namelist():
        if not f.endswith('/') and f.startswith('testDigits'):
            with z.open(f) as img_f:
                class_number = f.split('/')[-1].split('.')[0].split('_')[0]
                test_vect = img2vector(img_f)
                predict_class = classify(test_vect,train_vect,class_label,5)
                total_count += 1
                error_count += 0 if predict_class == class_number else 1
    print 'predict result'
    print 'total_count:%s' % total_count
    print 'error_count:%s' % error_count
    print 'error rate:%s' % (error_count/float(total_count))



if __name__=="__main__":
    run()