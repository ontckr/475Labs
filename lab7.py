import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import linear_model

with open("data/Grand-slams-men-2013-updated.csv") as f:
    teams_comb = list(csv.reader(f))

ACE1 = np.array([])
WNR1 = np.array([])
BPW1 = np.array([])
NPW1 = np.array([])
TPW1 = np.array([])

for rows in teams_comb:
    if rows != teams_comb[0]:
        ACE1 = np.append(ACE1, int(rows[4]))
        WNR1 = np.append(WNR1, int(rows[6]))
        BPW1 = np.append(BPW1, int(rows[9]))
        NPW1 = np.append(NPW1, int(rows[11]))
        TPW1 = np.append(TPW1, int(rows[12]))


first_column = np.ones((1, len(ACE1)))

X = np.vstack((first_column, ACE1, WNR1, BPW1, NPW1)).T
Y = np.vstack(TPW1)

# print(X)

lambda_values = np.arange(0, 10.1, 0.1)


last_fold = 0

MSE_values = []

for lamb in lambda_values:
    error = 0
    last_fold = 0
    for row in X:
        if (last_fold == 250):
            break
        X_test = X[last_fold: last_fold + 1, :]
        y_test = Y[last_fold: last_fold + 1, :]

        X_train = np.delete(X, slice(last_fold, last_fold + 1), axis=0)
        y_train = np.delete(Y, slice(last_fold, last_fold + 1), axis=0)
        # print(row)
        # print(y_test)

        lam = np.round(lamb, 2)
        lasso = linear_model.Lasso(alpha=lam)
        lasso.fit(X_train, y_train)

        prediction = lasso.predict(X_test)
        error += ((y_test - prediction)[0][0])**2
        last_fold += 1
    mse = error / len(X)
    MSE_values.append(mse)

# print(MSE_values)
minValue = min(MSE_values)
minIndex = MSE_values.index(minValue)
print("The lambda value", lambda_values[minIndex], "yields the minimum error of", minValue)

lasso = linear_model.Lasso(alpha=lambda_values[minIndex])
lasso.fit(X, Y)
print("The lasso coefficients with the optimal lambda value:")
print(lasso.coef_)

lasso = linear_model.Lasso(alpha=0)
lasso.fit(X, Y)
print("Regular least squares coefficients:")
print(lasso.coef_)

plt.plot(lambda_values, MSE_values)
plt.xlabel("Lambda Value")
plt.ylabel("Mean Squared Error")
plt.show()