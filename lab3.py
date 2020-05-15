import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/team.csv", encoding="Latin-1")

exp_arr = np.array(df['Experience'])
age_arr = np.array(df['Age'])
pow_arr = np.array(df['Power'])

x0 = np.ones(len(age_arr))

sal_arr = np.array(df['Salary'])

X = np.array((x0, exp_arr, age_arr, pow_arr)).T

# print(X)

x1 = X.T
x2 = np.dot(x1, x1.T)
x3 = np.linalg.inv(x2)
x4 = np.dot(x3, x1)

B = np.dot(x4, sal_arr)
Y = np.dot(X, B)
U = (Y - sal_arr)
U_abs = abs(U)

print(U_abs)

plt.scatter(Y, U_abs)
plt.show()
