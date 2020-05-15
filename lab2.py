import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv("data/team_1.csv", encoding="ISO-8859-1")
test = pd.read_csv("data/team_2.csv", encoding="ISO-8859-1")

age_list_1 = np.array(data["Age"])
exp_list_1 = np.array(data["Experience"])

age_list_2 = np.array(test["Age"])
exp_list_2 = np.array(test["Experience"])


def coef(x, y):
    totalX = 0
    totalY = 0
    for i in range(len(x)):
        totalX += x[i]
        totalY += y[i]

    mean_X = totalX / len(x)
    mean_Y = totalY / len(y)

    cov1 = 0
    cov2 = 0
    for i in range(0, len(x)):
        cov1 += ((x[i] - mean_X) * (y[i] - mean_Y))

    for i in range(0, len(x)):
        cov2 += (x[i] - mean_X) ** 2

    cov = cov1 / cov2
    m1 = cov
    b1 = (mean_Y) - ((mean_X) * m1)
    return [b1, m1]


def plot(x, y, c):
    X = np.linspace(np.min(x), np.max(x))
    Y = c[0] + (c[1] * X)
    plt.plot(X, Y, color='red')
    plt.scatter(x, y)
    plt.show()


def rss(x, y, c):
    rss_data = 0

    for r in range(len(y)):
        y_pred = c[0] + (c[1] * x[r])
        rss_data += (y[r] - y_pred) ** 2

    return rss_data


def tss(y):
    tss_data = 0
    y_sum = 0

    for i in range(len(y)):
        y_sum += y[i]

    y_avg = y_sum / len(y)

    for r in range(len(y)):
        tss_data += (y[r] - y_avg) ** 2

    return tss_data


def R2(rss,tss):
    return 1 - rss/tss


c = coef(age_list_1, exp_list_1)
ct = coef(age_list_2, exp_list_2)

plot(age_list_1, exp_list_1, ct)
plot(age_list_2, exp_list_2, c)

rss1 = rss(age_list_1, exp_list_1, ct)
rss2 = rss(age_list_2, exp_list_2, c)

tss1 = tss(exp_list_1)
tss2 = tss(exp_list_2)

print("R^2 score" , R2(rss1, tss1))
print("R^2 score" , R2(rss2, tss2))