import numpy as num
import csv
import math
import matplotlib.pyplot as plt

def dgi(y, y_pred):
    ss_cd = 0
    ss_as = 0

    # Compute ss_cd and ss_as for dgi
    for j in range(len(y)):
        ss_cd += y[j] * y[j] - y_pred[j] * y_pred[j]
        ss_as += (num.mean(y) - num.mean(y_pred)) * (num.mean(y) - num.mean(y_pred))
    se = num.sum(y - y_pred) ** 2
    res = se - (ss_cd + ss_as)

    
def compute_cubic_spline_regression(x, y, knots):

    x1 = x
    x2 = x ** 2
    x3 = x ** 3
    x4 = num.zeros((len(knots), len(x)))

    for i in range(len(x)):
        for k in range(len(knots)):
            if x1[i] > knots[k]:
                x4[k][i] = (x1[i] - knots[k]) ** 3

    ones = num.ones((1, len(x)))

    X = num.vstack((ones, x1, x2, x3, x4)).T

    y = y[X[:, 1].argsort()]
    X = X[X[:, 1].argsort()]

    coefficients = num.linalg.inv(num.dot(X.T, X))
    coefficients = num.dot(coefficients, X.T)
    coefficients = num.dot(coefficients, y)

    model = num.dot(X, coefficients)
    return model, X[:, 1]


if __name__ == '__main__':

    with open("Grand-slams-men-2013.csv") as f:
        csv_list = list(csv.reader(f))

    fsp1_list = num.array([])
    fsw1_list = num.array([])

    for row in csv_list:
        if row != csv_list[0]:
            fsp1_list = num.append(fsp1_list, int(row[6]))
            fsw1_list = num.append(fsw1_list, int(row[7]))

    dgi(fsp1_list, fsw1_list)
    knots = num.array([55, 65, 70])
    model_1, sorted_x = compute_cubic_spline_regression(fsp1_list, fsw1_list, knots)

    knots = num.array([60, 75])
    model_2, sorted_x = compute_cubic_spline_regression(fsp1_list, fsw1_list, knots)

    knots = num.array([62])
    model_3, sorted_x = compute_cubic_spline_regression(fsp1_list, fsw1_list, knots)

    plt.title("Cubic Spline Regression")
    plt.scatter(fsp1_list, fsw1_list)
    plt.plot(sorted_x, model_1, "g", label='3 knots')
    plt.plot(sorted_x, model_2, "r", label='2 knots')
    plt.plot(sorted_x, model_3, "k", label='1 knot')
    plt.xlabel("First serve percentage of player 1")
    plt.ylabel("First serve won by player 1")
    plt.legend()
    plt.show()
