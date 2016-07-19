# -*- coding: utf-8 -*-
__author__ = 'Administrator'


def create_tree(lenses, lenses_labels):
    class_list = [i[-1] for i in lenses]
    if class_list.count(class_list[0]) == len(class_list):  # 计算是否为同一种分类
        return class_list[0]
    if len(lenses[0]) == 1:
        return



def run():
    z = open("./bookdemo/Ch03/lenses.txt")
    lenses = [inst.strip().split('\t') for inst in z.readlines()]
    lenses_labels = ['age','prescript','astigmatic','tear_rate']
    lenses_tree = create_tree(lenses,lenses_labels)


if __name__=="__main__":
    run()