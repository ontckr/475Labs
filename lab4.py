import numpy as np
import csv
import matplotlib.pyplot as plt

def calculate_r2(y, y_prediction):

    tss = 0
    rss = 0

    for i in range(len(y)):
        tss += (y[i] - np.mean(y)) ** 2
        rss += (y[i] - y_prediction[i]) ** 2

    r2 = 1 - rss/tss
    print("R2 = " + str(r2))
    return r2

#read the file
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

last_fold = 0

cv_hat = []

y_hat = []

MSE = 0
MSE2 = 0

for i in range(10):
    #find 36 X data as name training data
    training_data = np.delete(X, slice(last_fold, last_fold + 4), axis=0)

    #find new Y(36)
    new_Y = np.delete(Y, slice(last_fold, last_fold + 4), axis=0)

    # print(training_data)
    # print("---------")

    test_data = X[last_fold: last_fold + 4, :]

    # print(test_data)
    # print("---------")

    #cross validations coefficients
    coefficient = np.linalg.inv(np.dot(training_data.T, training_data))
    coefficient = np.dot(coefficient, training_data.T)
    coefficient = np.dot(coefficient, new_Y)

    Y_prediction = np.dot(test_data, coefficient)
    # print(Y_prediction)

    cv_hat = np.append(cv_hat, Y_prediction)

    last_fold += 4


#without cross validation coefficients
coefficient = np.linalg.inv(np.dot(X.T, X))
coefficient = np.dot(coefficient, X.T)
coefficient = np.dot(coefficient, Y)

Y_prediction = np.dot(X, coefficient)

y_hat = np.append(y_hat, Y_prediction)

#croos validation mse
for i in range(len(cv_hat)):
    MSE += ((cv_hat[i] - Y[i]) ** 2) / len(cv_hat)


#without cross validation mse
for i in range(40):
    MSE2 += ((y_hat[i] - Y[i]) ** 2) / len(y_hat)


print("MSE with cross-validation: " + str(MSE))
print("MSE without cross-validation: " + str(MSE2))



# first graph

U = y_hat - Y  # without cv
U_abs = abs(U)
plt.scatter(y_hat, U_abs)

U2 = cv_hat - Y # with cv
U_abs2 = abs(U2)

plt.scatter(cv_hat, U_abs2)
plt.show()



# second graph

plt.scatter(y_hat, cv_hat)
plt.plot(y_hat,y_hat,'k-') #line
plt.show()