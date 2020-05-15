#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 12:25:43 2018

@author: diderot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../../Documents/Ce475Labs/lab5/Grand-slams-men-2013.csv", encoding="ISO-8859-1")

x = np.array(data["NPA.1"])
y = np.array(data["UFE.2"])
idx = x.argsort()
x = x[idx]
y = y[idx]
Y = y.transpose()

x0 = []
x2 = []
x3 = []

for i in range(0, len(x)):
    x0.append(1)

for i in range(0, len(x)):
    x2.append(x[i] ** 2)

for i in range(0, len(x)):
    x3.append(x[i] ** 3)


def Xknot1(knotValue1):
    knot1 = []
    for i in range(0, len(x)):
        if x[i] - knotValue1 > 0:
            knot1.append((x[i] - knotValue1) ** 3)
        else:
            knot1.append(0)
    X1 = np.array([x0, x, x2, x3, knot1])

    X1_t = X1.transpose()
    return X1_t


def Xknot2(knotValue1, knotValue2):
    knot2 = []
    knot1 = []
    for i in range(0, len(x)):
        if x[i] - knotValue1 > 0:
            knot1.append((x[i] - knotValue1) ** 3)
        else:
            knot1.append(0)
    for i in range(0, len(x)):
        if x[i] - knotValue2 > 0:
            knot2.append((x[i] - knotValue2) ** 3)
        else:
            knot2.append(0)
    X2 = np.array([x0, x, x2, x3, knot1, knot2])

    X2_t = X2.transpose()
    return X2_t


def Xknot3(knotValue1, knotValue2, knotValue3):
    knot3 = []
    knot2 = []
    knot1 = []
    for i in range(0, len(x)):
        if x[i] - knotValue1 > 0:
            knot1.append((x[i] - knotValue1) ** 3)
        else:
            knot1.append(0)
    for i in range(0, len(x)):
        if x[i] - knotValue2 > 0:
            knot2.append((x[i] - knotValue2) ** 3)
        else:
            knot2.append(0)

    for i in range(0, len(x)):
        if x[i] - knotValue3 > 0:
            knot3.append((x[i] - knotValue3) ** 3)
        else:
            knot3.append(0)

    X3 = np.array([x0, x, x2, x3, knot1, knot2, knot3])

    X3_t = X3.transpose()
    return X3_t


def betaPred(X):
    firstDot = (np.dot(X.transpose(), X))
    inBeta = np.linalg.inv(firstDot)
    prodXt = np.dot(inBeta, X.transpose())
    yDot = np.dot(prodXt, Y)
    return yDot


def predict_Y(x, b):
    return np.dot(x, b)


knot1 = predict_Y(Xknot1(30), betaPred(Xknot1(30)))
knot2 = predict_Y(Xknot2(15, 30), betaPred(Xknot2(15, 30)))
knot3 = predict_Y(Xknot3(10, 20, 30), betaPred(Xknot3(10, 20, 30)))
plt.scatter(x, y)
plt.plot(x, knot1, color="black")
plt.plot(x, knot2, color="red")
plt.plot(x, knot3, color="green")
plt.show()
print(betaPred(Xknot2(15, 30)))
print(knot2)
