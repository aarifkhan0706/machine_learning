# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 00:19:19 2019

@author: AARIF KHAN
"""
#implementing my first machine learning via matplotlib,svm
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
digits=datasets.load_digits()
print(digits.data)
print(digits.target)
print(digits.images[0])
