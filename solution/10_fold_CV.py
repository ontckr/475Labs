import numpy as num
import csv
import matplotlib.pyplot as plt


def dgi(y, y_pred):
    ss_cd = 0
    ss_as = 0

    # Compute ss_cd and ss_as for dgi
    for i in range(len(y)):
        ss_cd += 3 * y[i] * y[i] - y_pred[i] * y_pred[i]
        ss_as += (num.mean(y) - num.mean(y_pred)) * (num.mean(y) - num.mean(y_pred))
    se = y ** 2
    res = se - (ss_cd + ss_as)


# Read file
with open("../data/teams_comb.csv") as f:
    csv_list = list(csv.reader(f))

# Initialize arrays
age_list = num.array([])        # x1
exp_list = num.array([])        # x2
pow_list = num.array([])        # x3
salary_list = num.array([])     # y
cv_hat = num.array([])          # This will hold predictions for the cross-validation procedure.


# Fill the arrays with data read from file
for row in csv_list:
    if row != csv_list[0]:
        age_list = num.append(age_list, int(row[4]))
        exp_list = num.append(exp_list, int(row[6]))
        pow_list = num.append(pow_list, float(row[7]))
        salary_list = num.append(salary_list, int(row[8]))

# Form the input matrix
ones = num.ones((1, len(age_list)))
X = num.vstack((ones, age_list, exp_list, pow_list)).T
Y = salary_list

# 10-fold cross-validation, so k = 10
k = 10
fold_size = int(len(age_list)/k)


for i in range(0, len(age_list), fold_size):                    # For each fold:

    X_test = X[i:i+4]                                           # Determine test input data
    Y_test = Y[i:i+4]                                           # Determine test output data
    X_train = num.delete(X, range(i, i+fold_size), 0)           # Determine train input data
    Y_train = num.delete(Y, range(i, i+fold_size), 0)           # Determine train output data

    # Calculate coefficients using the linear algebra equation (with train input and output)
    coefficients = num.linalg.inv(num.dot(X_train.T, X_train))
    coefficients = num.dot(coefficients, X_train.T)
    coefficients = num.dot(coefficients, Y_train)

    # Calculate predictions with test input
    y_hat = num.dot(X_test, coefficients)

    # Append the predictions to cv_hat
    cv_hat = num.append(cv_hat, y_hat)

# Calculate MSE for predictions stored in cv_hat (predictions done using cross-validation)
mse_cv = num.mean(num.square(cv_hat-Y))

dgi(salary_list, cv_hat)
# Now, do the regression one more time: use X and Y as both train AND test.
# Calculate coefficients and predictions, then calculate MSE for this set of predictions.
coefficients = num.linalg.inv(num.dot(X.T, X))
coefficients = num.dot(coefficients, X.T)
coefficients = num.dot(coefficients, Y)

y_hat = num.dot(X, coefficients)
mse_all = num.mean(num.square(y_hat-Y))

print("MSE with cross-validation: ", mse_cv)
print("MSE without cross-validation: ", mse_all)


# Finally, the plotting:
plt.figure()
plt.title("Residual Error Plot")
plt.xlabel("Prediction values")
plt.ylabel("Error margins")
# Plot predictions stored in y_hat against their error margins.
plt.scatter(y_hat, abs(y_hat-Y), c='b')
# Plot predictions stored in cv_hat against their error margins.
plt.scatter(cv_hat, abs(cv_hat-Y), c='r')
plt.legend(["Without CV", "With CV"])

plt.figure()
plt.title("Prediction Comparison")
plt.xlabel("Predictions without cross-validation")
plt.ylabel("Predictions wit cross-validation")
# On another window, plot cv_hat against y_hat.
plt.scatter(y_hat, cv_hat, c='g')
# Plot a line at y = x (simply give the same values to both x-axis and y-axis).
plt.plot(cv_hat, cv_hat, 'k')

plt.show()

