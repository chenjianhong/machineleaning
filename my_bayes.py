# -*- coding: utf-8 -*-
'''
朴素贝叶斯分类器：
优点：
    1.数据较少时仍然有效,可以处理多类别问题。
缺点：
    1.对于输入数据的准备方式比较敏感
适用数据类型:
    标称型
'''
import numpy


def load_data_set():
    feature_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_list = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return feature_list, class_list

def create_vocab_list(data_set):
    vocab_set = set()
    for d in data_set:
        vocab_set |= set(d)
    return list(vocab_set)

def word_to_vocab_vec(vocab_list,input_set):
    word_vec = [0]*len(vocab_list)
    for w in input_set:
        if w in vocab_list:
            word_vec[vocab_list.index(w)] = 1
    return word_vec

def train_nb(feature_matrix,label_matrix):
    item_count = len(feature_matrix)
    vocab_count = len(feature_matrix[0])
    p1_total_proba = sum(label_matrix)/float(item_count)
    p1_feature_count = numpy.ones(vocab_count)
    p0_feature_count = numpy.ones(vocab_count)
    p1_denom = 2.0
    p0_denom = 2.0
    for i in range(item_count):
        if int(label_matrix[i]) == 1:
            p1_feature_count += feature_matrix[i]
            p1_denom += sum(feature_matrix[i])
        else:
            p0_feature_count += feature_matrix[i]
            p0_denom += sum(feature_matrix[i])
    p1_proba = numpy.log(p1_feature_count/p1_denom)
    p0_proba = numpy.log(p0_feature_count/p0_denom)
    return p1_proba,p0_proba,p1_total_proba

def classify_nb(this_doc,p1_proba,p0_proba,p1_total_proba):
    p1 = sum(this_doc*p1_proba) + numpy.log(p1_total_proba)
    p0 = sum(this_doc*p0_proba) + numpy.log(1.0 - p1_total_proba)
    if p1 > p0:
        return 1
    else:
        return 0


def run():
    word_list, class_list = load_data_set()
    vocab_list = create_vocab_list(word_list)
    feature_list = list()
    for p in word_list:
        feature_list.append(word_to_vocab_vec(vocab_list,p))
    p1_proba, p0_proba, p1_total_proba = train_nb(numpy.array(feature_list),numpy.array(class_list))
    test_entry = ['love','my','dalmation']
    # test_entry = ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid']
    # test_entry = ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please']
    this_doc = word_to_vocab_vec(vocab_list,test_entry)
    print 'test label is:%s' % classify_nb(this_doc,p1_proba,p0_proba,p1_total_proba)

if __name__=="__main__":
    run()