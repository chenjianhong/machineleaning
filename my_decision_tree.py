# -*- coding: utf-8 -*-
'''
python list 的count用法：计算列表项的出现次数
>>> a = [1,1,1,1,2]
>>> a.count(1)

'''
import operator

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
    pass


def split_data_set(data_set, i, v):
    pass


def choose_best_feature_to_split(data_set):
    num_features = len(data_set) - 1
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



def run():
    z = open("./bookdemo/Ch03/lenses.txt")
    lenses = [inst.strip().split('\t') for inst in z.readlines()]
    lenses_labels = ['age','prescript','astigmatic','tear_rate']
    lenses_tree = create_tree(lenses,lenses_labels)


if __name__=="__main__":
    run()