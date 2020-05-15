import matplotlib.pyplot as plt
import numpy as np
import csv

# read data
with open("data/Grand-slams-men-2013.csv") as f:
    teams_comb = list(csv.reader(f))

x = np.array([])
y = np.array([])

for rows in teams_comb:
    if rows != teams_comb[0]:
        x = np.append(x, int(rows[6]))
        y = np.append(y, int(rows[7]))


def cubic_reg(x, y, knots):
    sorted_x = x.argsort()

    x = x[sorted_x]
    y = y[sorted_x]

    first_column = np.ones((len(x), 1))  # first 1's column

    x1 = np.array([x]).T
    x2 = x1 ** 2
    x3 = x1 ** 3

    knot_columns = []

    for knot in knots:
        final = []
        for a in x1.flatten():
            result = a - knot

            if (result < 0):  # check the results
                result = 0
                final.append(result)
            else:
                final.append(result)

        column = np.array([final]).T
        column3 = column ** 3
        knot_columns.append(column3)

    knot_size = len(knot_columns)  # knot length

    if knot_size == 1:
        X = np.hstack((first_column, x1, x2, x3, knot_columns[0]))
    elif knot_size == 2:
        X = np.hstack((first_column, x1, x2, x3, knot_columns[0], knot_columns[1]))
    elif knot_size == 3:
        X = np.hstack((first_column, x1, x2, x3, knot_columns[0], knot_columns[1], knot_columns[2]))

    B1 = np.dot(X.T, X)
    B2 = np.linalg.inv(B1).dot(X.T)
    coefficient = np.dot(B2, y)
    regression = X.dot(coefficient)

    return x, y, regression


reg1 = cubic_reg(x, y, [55, 65, 70])
plt.plot(reg1[0], reg1[2], color="green")

reg2 = cubic_reg(x, y, [60, 75])
plt.plot(reg2[0], reg2[2], color="red")

reg3 = cubic_reg(x, y, [62])
plt.plot(reg3[0], reg3[2], color="black")

plt.scatter(reg1[0], reg1[1], color="blue")
plt.show()
