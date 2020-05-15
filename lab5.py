import numpy as np
import csv


def R2(y, y_prediction):
    tss = 0
    rss = 0

    for i in range(len(y)):
        tss += (y[i] - np.mean(y)) ** 2
        rss += (y[i] - y_prediction[i]) ** 2

    r2 = 1 - rss / tss

    print("R2 = " + str(r2))
    return r2


#   R2 adjust method

def R2_adjust(y, y_prediction, d):
    tss = 0
    rss = 0

    for i in range(len(y)):
        tss += (y[i] - np.mean(y)) ** 2
        rss += (y[i] - y_prediction[i]) ** 2

    r2 = 1 - ((rss / ((len(y)) - d - 1)) / (tss / (len(y) - 1)))

    print("R2 adjust = " + str(r2))
    return r2


def simplelinear_coef(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    ss_xy, ss_xx = 0, 0

    for i in range(n):
        ss_xy += (x[i] - mean_x) * (y[i] - mean_y)
        ss_xx += np.square(x[i] - mean_x)

    b_1 = ss_xy / ss_xx
    b_0 = mean_y - (b_1 * mean_x)

    return [b_0, b_1]


# read the file
with open("data/teams_comb.csv") as f:
    teams_comb = list(csv.reader(f))

age_list = np.array([])
experience_list = np.array([])
power_list = np.array([])
salary_list = np.array([])

for rows in teams_comb:
    if rows != teams_comb[0]:
        age_list = np.append(age_list, int(rows[4]))
        experience_list = np.append(experience_list, int(rows[6]))
        power_list = np.append(power_list, float(rows[7]))
        salary_list = np.append(salary_list, int(rows[8]))

fist_column = np.ones((1, len(age_list)))

X = np.vstack((fist_column, age_list, experience_list, power_list)).T
Y = salary_list.T


# print(X)
# print(Y)


def expression(a, b, list):
    return a + b * list


def M0(y):
    mean_y = sum(y) / len(y)
    q = np.vstack(fist_column).T   # 1's column matrix
    return q * mean_y


A = M0(Y)
R2_adjust(Y, A, 0)  # first adjust R2 score


def M1(X, Y):
    [x1, x2] = simplelinear_coef(X, Y)
    # print(x1, x2)
    return x1, x2


array = [age_list, experience_list, power_list]     # for using loop

for element in array:
    b0, b1 = M1(element, Y)
    y_pred = expression(b0, b1, element)
    r2 = R2(Y, y_pred)

b0, b1 = M1(power_list, Y)
y_pred = expression(b0, b1, power_list)

R2_adjust(Y, y_pred, 1)  # second adjust R2 score