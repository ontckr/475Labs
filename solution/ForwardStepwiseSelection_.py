import numpy as num
import csv
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


def adjusted_r2_score(y, y_pred, d):

    n = len(y)
    rss = 0
    tss = 0
    for i in range(len(y)):
        rss += (y[i]-y_pred[i])**2
        tss += (y[i]-num.mean(y))**2

    adjusted_r2 = 1 - (rss / (n-d-1))/(tss / (n-1))
    return adjusted_r2


def r2_score(y, y_pred):

    rss = 0
    tss = 0
    for i in range(len(y)):
        rss += (y[i]-y_pred[i])**2
        tss += (y[i]-num.mean(y))**2

    r_squared = 1 - rss/tss
    return r_squared,


def fit_linear(X_train, y_train, X_test):

    coefficients = num.linalg.inv(num.dot(X_train.T, X_train))
    coefficients = num.dot(coefficients, X_train.T)
    coefficients = num.dot(coefficients, y_train)

    predictions = num.dot(X_test, coefficients)
    return predictions


# Read file
with open("teams_comb.csv") as f:
    csv_list = list(csv.reader(f))

# Initialize arrays
age_list = num.array([])        # x1
exp_list = num.array([])        # x2
pow_list = num.array([])        # x3
salary_list = num.array([])     # y
titles = num.array([])          # Array to hold titles.


# Fill the arrays with data read from file
for row in csv_list:
    if row != csv_list[0]:
        age_list = num.append(age_list, int(row[4]))
        exp_list = num.append(exp_list, int(row[6]))
        pow_list = num.append(pow_list, float(row[7]))
        salary_list = num.append(salary_list, int(row[8]))
    else:
        titles = num.append(titles, row[4])
        titles = num.append(titles, row[6])
        titles = num.append(titles, row[7])

# Form the input matrix
ones = num.ones(len(age_list))
X = num.vstack((ones, age_list, exp_list, pow_list)).T
Y = salary_list

dgi(salary_list, age_list)
r2_scores = num.array([])
means = ones * num.mean(salary_list)

# The first adjusted R^2 score is 0, since we have no models and RSS = TSS.
M_0 = adjusted_r2_score(salary_list, means, 0)  # This should be 0.

for i in range(1, len(X[0])):

    # For x1, x2 and x3, we test R^2 scores. We build matrices and create predictions using training data.
    X_temp = num.vstack((X[:, 0], X[:, i])).T
    y_pred = fit_linear(X_temp, salary_list, X_temp)
    r2_scores = num.append(r2_scores, r2_score(salary_list, y_pred))

idx = num.argmax(r2_scores)
X_temp = num.vstack((X[:, 0], X[:, idx+1])).T
y_pred = fit_linear(X_temp, salary_list, X_temp)
M_1 = adjusted_r2_score(salary_list, y_pred, 1)

print(titles[idx] + " has been shown to yield the best R^2 score.")
print("First adjusted R^2 score: " + str(M_0))
print("Second adjusted R^2 score: " + str(M_1))

