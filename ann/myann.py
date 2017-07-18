# -*- coding: utf-8 -*-
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn import datasets

def generate_data():
    np.random.seed(0)
    x,y = datasets.make_moons(200,noise=0.20)
    plt.scatter(x[:,0],x[:,1],s=20,c=y)
    plt.show()


if __name__=="__main__":
    generate_data()