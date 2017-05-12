# -*- coding: utf-8 -*-
'''
k 近邻算法：
优点：
    1.计算复杂度不高,输出结果易于理解,对中间值的缺失不敏感,可以处理不相关特征
缺点：
    1.可能产生过拟合
适用数据类型:
    数值型和标称型
程序计算逻辑：
    1.创建决策树
        1)根据剩余特征的熵,确认熵减少幅度最大的特征
        2)该特征的每个特征值都作为一个分支,然后重复1步骤,直到特征用尽或者只剩一个类别
'''
import copy
import operator

import math

__author__ = 'Administrator'


def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count:
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sorted_class_count[0][0]


def calc_shannon_ent(data_set):
    data_count = len(data_set)
    label_count = dict()
    for f in data_set:
        current_label = f[-1]
        if current_label not in label_count: label_count[current_label] = 0
        label_count[current_label] += 1
    shannon_ent = 0
    for l in label_count:
        prob = label_count[l]*1.0/data_count
        shannon_ent -= prob * math.log(prob,2)
    return shannon_ent


def split_data_set(data_set, i, v):
    split_result = list()
    for d in data_set:
        if d[i] == v:
            reduced_feature = d[:i]
            reduced_feature.extend(d[i+1:])
            split_result.append(reduced_feature)
    return split_result


def choose_best_feature_to_split(data_set):
    num_features = len(data_set[0]) - 1
    base_entropy = calc_shannon_ent(data_set)
    best_info_gain = 0
    best_feature = -1
    for i in range(num_features):
        feat_list = [j[i] for j in data_set]
        unique_vals = set(feat_list)
        new_entropy = 0
        for v in unique_vals:
            sub_data_set = split_data_set(data_set,i,v)
            prob = len(sub_data_set)/float(len(data_set))
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def create_tree(lenses, lenses_labels):
    class_list = [i[-1] for i in lenses]
    if class_list.count(class_list[0]) == len(class_list):  # 计算是否为同一种分类
        return class_list[0]
    if len(lenses[0]) == 1: # 当只有一个特征时停止分割
        return majority_cnt(class_list)
    best_feat = choose_best_feature_to_split(lenses)
    best_feat_label = lenses_labels[best_feat]
    my_tree = {best_feat_label:{}}
    del lenses_labels[best_feat]
    for f_v in set([i[best_feat] for i in lenses]):
        sub_labels = lenses_labels[:]
        my_tree[best_feat_label][f_v] = create_tree(split_data_set(lenses,best_feat,f_v),sub_labels)
    return my_tree

def classify(input_tree,feat_labels,test_vec):
    first_feat = input_tree.keys()[0]
    second_dict = input_tree[first_feat]
    feat_index = feat_labels.index(first_feat)
    key = test_vec[feat_index]
    value_of_feat = second_dict[key]
    if isinstance(value_of_feat,dict):
        class_label = classify(value_of_feat,feat_labels,test_vec)
    else:
        class_label = value_of_feat
    return class_label



def run():
    z = open("./bookdemo/Ch03/lenses.txt")
    lenses = [inst.strip().split('\t') for inst in z.readlines()]
    lenses_labels = ['age','prescript','astigmatic','tear_rate']
    lenses_tree = create_tree(lenses,copy.deepcopy(lenses_labels))
    print lenses_tree
    print '*'*100
    print 'predict label:%s' % classify(lenses_tree,lenses_labels,['presbyopic','myope','yes','normal'])


if __name__=="__main__":
    run()