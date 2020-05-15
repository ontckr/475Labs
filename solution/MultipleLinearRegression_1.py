import numpy as num
import csv
import math
import matplotlib.pyplot as plt

# Multiple Linear Regression for LAB 3

def rotation_min_find_dac(a):

    first = a[0]
    last = a[len(a) - 1]

    if len(a) == 1:
        return first

    if len(a) == 2 and first == last:
        return first

    if first < last:
        return first

    pivot = math.floor(len(a)/2)

    if a[pivot] < a[pivot-1]:
        return a[pivot]
    elif a[pivot] < first:
        return rotation_min_find_dac(a[0:pivot])
    else:
        return rotation_min_find_dac(a[pivot+1:len(a)])

# Function to calculate R^2 score as we did in the previous lab.
def r2_score(y, y_pred):

    rss = 0
    tss = 0
    for i in range(len(y)):
        rss += (y[i]-y_pred[i])**2
        tss += (y[i]-num.mean(y))**2

    r_squared = 1 - rss/tss
    print("R^2 score: " + str(r_squared))


with open("../data/team.csv") as f:
    csv_list = list(csv.reader(f))

age_list = num.array([])
exp_list = num.array([])
pow_list = num.array([])
salary_list = num.array([])

# Extracting data into lists, creating X and y:

for row in csv_list:
    if row != csv_list[0]:
        age_list = num.append(age_list, int(row[4]))
        exp_list = num.append(exp_list, int(row[6]))
        pow_list = num.append(pow_list, float(row[7]))
        salary_list = num.append(salary_list, int(row[8]))

# Forming the input(X) and the output(y)
ones = num.ones((1, len(age_list)))
X = num.vstack((ones, age_list, exp_list, pow_list)).T
y = salary_list

# Calculating coefficients using the linear algebra equation:
coefficients = num.linalg.inv(num.dot(X.T, X))
coefficients = num.dot(coefficients, X.T)
coefficients = num.dot(coefficients, y)

# Calculating the regression line (estimations):
y_hat = num.dot(X, coefficients)

# Calculating and displaying the R^2 score for this set of predictions:
print("Showing original results:")
r2_score(y, y_hat)

# Now, the next part: create an array of random integers from -500 to 500, having the same length as our number of rows
random_cols = num.random.randint(-500, 500, len(age_list))

# Add the random column to our input matrix X. This can also be done using other stacking functions (stack, hstack etc.)
X = num.vstack((X.T, random_cols)).T

# Calculating coefficients using the linear algebra equation once more:
coefficients = num.linalg.inv(num.dot(X.T, X))
coefficients = num.dot(coefficients, X.T)
coefficients = num.dot(coefficients, y)

# Calculating the regression line (estimations) once more:
y_hat = num.dot(X, coefficients)

# Calculating and displaying the R^2 score for this set of predictions:
print("Showing results with an added random column:")
r2_score(y, y_hat)

# Plotting estimation vs error:
# Scatter plot with predictions as our x-axis, and errors as our y-axis.
# Errors are calculated simply by subtracting original results from their corresponding predictions,
#   and taking the absolute value of the result.
plt.title("Residual Error Plot")
plt.scatter(y_hat, num.abs(y-y_hat))
plt.show()
