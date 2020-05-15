import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

# with open("team.csv",encoding="Latin-1") as f:
#     csv_list = list(csv.reader(f))
#
# x1 = []
# x2 = []
# x3 = []
# y = []
#
# for player in csv_list:
#     if player != csv_list[0]:
#         age = int(player[4])
#         exp = int(player[6])
#         pow = float(player[7])
#         sal = int(player[8])
#
#         x1.append(age)
#         x2.append(exp)
#         x3.append(pow)
#         y.append(sal)
#
#
# x0 = np.ones(len(x1), dtype=int)
#
# X = np.array([x0, x1, x2, x3]).T
#
# print(X)
#
# print(y)


team = pd.read_csv('data/team.csv', encoding='latin-1')
x_team = team.iloc[:,4].values
y_team = team.iloc[:,6].values
z_team = team.iloc[:,7].values
S_team = team.iloc[:,8].values
x_team = x_team.reshape(-1,1)
y_team = y_team.reshape(-1,1)
z_team = z_team.reshape(-1,1)
S_team = S_team.reshape(-1,1)

matx=np.concatenate([x_team.T,y_team.T,z_team.T])

X=matx.T
X=np.append(arr=np.ones((18,1)).astype(int),values=X,axis=1)
print(X)
print(S_team)


a1 = X.T
a2 = np.dot(a1, a1.T)
a3 = np.linalg.inv(a2)
a4 = np.dot(a3, a1)
B = np.dot(a4, S_team)

Y = np.dot(X, B)

U = Y - S_team
print(U)

plt.scatter(Y, U)
plt.hlines(y=0, xmin = 0, xmax = 20000)
plt.show()