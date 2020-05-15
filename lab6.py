import numpy as np
import csv
import matplotlib.pyplot as plt

# read the file
with open("data/Grand-slams-men-2013.csv") as f:
    teams_comb = list(csv.reader(f))

fsp = np.array([])
fsw = np.array([])

for rows in teams_comb:
    if rows != teams_comb[0]:
        fsp = np.append(fsp, int(rows[6]))
        fsw = np.append(fsw, int(rows[7]))

X = fsp
Y = fsw

# print(X)
print(Y)

X2 = X ** 2
X3 = X ** 3

first_column = np.ones((1, len(X)))


def knot_single(value1):

    knot1 = []

    for i in range(0, len(X)):
        if X[i] - value1 > 0:
            knot1.append((X[i] - value1) ** 3)
        else:
            knot1.append(0)

    single_final = np.vstack((first_column, X, X2, X3, knot1)).T
    return single_final


def knot_double(value1, value2):

    knot1 = []
    knot2 = []

    for i in range(0, len(X)):
        if X[i] - value1 > 0:
            knot1.append((X[i] - value1) ** 3)
        else:
            knot1.append(0)

    for i in range(0, len(X)):
        if X[i] - value2 > 0:
            knot2.append((X[i] - value2) ** 3)
        else:
            knot2.append(0)

    double_final = np.vstack((first_column, X, X2, X3, knot1, knot2)).T
    return double_final


def knot_triple(value1, value2, value3):

    knot1 = []
    knot2 = []
    knot3 = []

    for i in range(0, len(X)):
        if X[i] - value1 > 0:
            knot1.append((X[i] - value1) ** 3)
        else:
            knot1.append(0)

    for i in range(0, len(X)):
        if X[i] - value2 > 0:
            knot2.append((X[i] - value2) ** 3)
        else:
            knot2.append(0)

    for i in range(0, len(X)):
        if X[i] - value3 > 0:
            knot3.append((X[i] - value3) ** 3)
        else:
            knot3.append(0)

    triple_final = np.vstack((first_column, X, X2, X3, knot1, knot2, knot3)).T
    return triple_final
