import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sys
np.set_printoptions(threshold=sys.maxsize)

with open("data/Grand-slams-men-2013.csv") as f:
    teams_comb = list(csv.reader(f))

FSP1 = np.array([])
ACE1 = np.array([])
DBF1 = np.array([])
WNR1 = np.array([])
UFE1 = np.array([])
BPC1 = np.array([])
NPA1 = np.array([])

for rows in teams_comb:
    if rows != teams_comb[0]:
        FSP1 = np.append(FSP1, int(rows[6]))
        ACE1 = np.append(ACE1, int(rows[10]))
        DBF1 = np.append(DBF1, int(rows[11]))
        WNR1 = np.append(WNR1, int(rows[12]))
        UFE1 = np.append(UFE1, int(rows[13]))
        BPC1 = np.append(BPC1, int(rows[14]))
        NPA1 = np.append(NPA1, int(rows[16]))

X = np.vstack((FSP1, ACE1, DBF1, WNR1, UFE1, BPC1, NPA1)).T

# print(X)

ST1_1 = np.array([])
ST2_1 = np.array([])
ST3_1 = np.array([])
ST4_1 = np.array([])
ST5_1 = np.array([])

for rows in teams_comb:
    if rows != teams_comb[0]:
        ST1_1 = np.append(ST1_1, int(rows[19]))
        ST2_1 = np.append(ST2_1, int(rows[20]))
        ST3_1 = np.append(ST3_1, int(rows[21]))
        ST4_1 = np.append(ST4_1, int(rows[22]))
        ST5_1 = np.append(ST5_1, int(rows[23]))

Y = ST1_1 + ST2_1 + ST3_1 + ST4_1 + ST5_1

X_Train = X[0:200]
X_Test = X[200:240]
# print(X_Train)
# print(X_Test)
Y_Train = Y[0:200]
Y_Test = Y[200:240]
# print(Y_Test)
# print(Y_Train)


def R2(y, y_prediction):
    tss = 0
    rss = 0

    for i in range(len(y)):
        tss += (y[i] - np.mean(y)) ** 2
        rss += (y[i] - y_prediction[i]) ** 2
    r2 = 1 - rss / tss
    return r2


R2_1 = []
R2_2 = []
R2_3 = []

for i in range(1, 151):

    regression_a = RandomForestRegressor(max_features="auto", max_depth=7, n_estimators=i)
    regression_b = RandomForestRegressor(max_features="sqrt", max_depth=7, n_estimators=i)
    regression_c = RandomForestRegressor(max_features=4, max_depth=7, n_estimators=i)

    regression_a.fit(X_Train, Y_Train)
    regression_b.fit(X_Train, Y_Train)
    regression_c.fit(X_Train, Y_Train)

    y_pred_a = regression_a.predict(X_Test)
    y_pred_b = regression_b.predict(X_Test)
    y_pred_c = regression_c.predict(X_Test)

    R2_1.append(R2(Y_Test, y_pred_a))
    R2_2.append(R2(Y_Test, y_pred_b))
    R2_3.append(R2(Y_Test, y_pred_c))


regression1 = RandomForestRegressor(max_features=4, max_depth=7, n_estimators=150)
regression2 = RandomForestRegressor(max_features=4, max_depth=1, n_estimators=150)

regression1.fit(X_Train, Y_Train)
regression2.fit(X_Train, Y_Train)

y_pred_1 = regression1.predict(X_Test)
y_pred_2 = regression2.predict(X_Test)

error1 = Y_Test - y_pred_1
error2 = Y_Test - y_pred_2


X_value = np.arange(1, 151, 1)

fig, ax = plt.subplots()
ax.plot(X_value, R2_1, 'g', label="m = p")
ax.plot(X_value, R2_2, 'b', label="m = sqrt(p)")
ax.plot(X_value, R2_3, 'r', label="m = p/2")

plt.xlabel("Number of estimators (decision trees)")
plt.ylabel("R^2 score")

legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')
plt.show()


plt.scatter(y_pred_1, error1, color='r')
plt.scatter(y_pred_2, error2, color='b')
plt.xlabel("Estimation")
plt.ylabel("Error of estimation")
plt.show()
