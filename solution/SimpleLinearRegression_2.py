import csv
import numpy as num
import matplotlib.pyplot as plt
import math


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


# Calculating coefficients based on the equation given in the instructions
def simlin_coef(x, y):

    n = len(x)
    mean_x = sum(x) / n     # Alternative: mean_x = num.mean(x)
    mean_y = sum(y) / n     # Alternative: mean_y = num.mean(y)
    ss_xy, ss_xx = 0, 0

    for i in range(n):
        ss_xy += (x[i] - mean_x) * (y[i] - mean_y)
        ss_xx += num.square(x[i] - mean_x)

    b_1 = ss_xy / ss_xx
    b_0 = mean_y - (b_1 * mean_x)

    return [b_0, b_1]


# Function for plotting
def simlin_plot(x, y, reg):

    # Open a new blank window (not necessary if you only have a single window for the entire program)
    plt.figure()

    # Put title and labels for axes
    plt.title("Simple Linear Regression: Age vs Experience")
    plt.xlabel("Age")
    plt.ylabel("Experience")

    # Scatter plot for the data points, x vs y
    # The marker is "o" by default, so this doesn't do anything. You can try changing it to other markers, e.g. "x".
    plt.scatter(x, y, marker="o")

    # Line plot for the regression line
    # Third parameter is for the color, red in this case.
    plt.plot(x, reg, "r")


# Function to calculate R^2 score, using the equations given in the instructions.
def simlin_r2(y, y_pred):

    rss = 0
    tss = 0

    for i in range(len(y)):
        rss += (y[i]-y_pred[i])**2
        tss += (y[i]-num.mean(y))**2

    r_squared = 1 - rss/tss

    print("R^2 score: " + str(r_squared))


# ---------------------------------- Script starts here -----------------------------------------

# Read the first .csv file into a list
with open("../data/team_1.csv") as f:
    csv_list_1 = list(csv.reader(f))

# Initialize empty arrays to fill in later
age_list_1 = num.array([])
exp_list_1 = num.array([])

# Read the second .csv file into a list
with open("../data/team_2.csv") as f:
    csv_list_2 = list(csv.reader(f))

# Initialize empty arrays to fill in later
age_list_2 = num.array([])
exp_list_2 = num.array([])

# Filling in the arrays
for row in csv_list_1:
    if row != csv_list_1[0]:
        age_list_1 = num.append(age_list_1, int(row[4]))
        exp_list_1 = num.append(exp_list_1, int(row[6]))

for row in csv_list_2:
    if row != csv_list_2[0]:
        age_list_2 = num.append(age_list_2, int(row[4]))
        exp_list_2 = num.append(exp_list_2, int(row[6]))


# Calculate coefficients
[b1, m1] = simlin_coef(age_list_1, exp_list_1)
[b2, m2] = simlin_coef(age_list_2, exp_list_2)

# Calculate regression lines as given in the instructions.
# Use one set of inputs with coefficients calculated using the other set of inputs, and vice versa.
exp_pred_1 = b2 + m2 * age_list_1
exp_pred_2 = b1 + m1 * age_list_2

# Calculate R^2 scores
simlin_r2(exp_list_1, exp_pred_1)
simlin_r2(exp_list_2, exp_pred_2)

# Draw plots
simlin_plot(age_list_1, exp_list_1, exp_pred_1)
simlin_plot(age_list_2, exp_list_2, exp_pred_2)

# Show plots
plt.show()
