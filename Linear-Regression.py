######################################################
# Sales Prediction with Linear Regression
######################################################

# I have developed a sales forecasting model. This model indicates the amount of sales obtained as a result of advertising expenses made on various channels.
# My first task is to build a simple regression model with two variables, and then proceed to build a model with all five variables present in the dataset.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

######################################################
# Simple Linear Regression with OLS Using Scikit-Learn
######################################################

df = pd.read_csv("C:/Users/Emre Bilir/Desktop/DSBootcamp/machine_learning/datasets/advertising.csv")

# In order to grasp the basic logic of regression, I will proceed by selecting two variables from this dataframe.
X = df[["TV"]]
y = df[["sales"]]

##########################
# Model
##########################

reg_model = LinearRegression().fit(X, y)

# Intercept (b - bias)
reg_model.intercept_[0]

# Coefficient of TV (w1)
reg_model.coef_[0][0]

# Visualization of the Model
g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},
                ci=False, color="r")

g.set_title(f"Model Equation: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Number of Sales")
g.set_xlabel("TV Expenses")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show(block=True)

##########################
# Prediction Accuracy
##########################

# MSE
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)
# 10.512652915656757

# RMSE
np.sqrt(mean_squared_error(y, y_pred))
# 3.2423221486546887

# MAE
mean_absolute_error(y, y_pred)
# 2.549806038927486

# R-squared
reg_model.score(X, y)
# 0.611875050850071

######################################################
# Multiple Linear Regression
######################################################

df = pd.read_csv("C:/Users/Emre Bilir/Desktop/DSBootcamp/machine_learning/datasets/advertising.csv")

X = df.drop('sales', axis=1)
y = df[["sales"]]

##########################
# Model
##########################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

reg_model = LinearRegression().fit(X_train,y_train)

# sabit (b - bias)
reg_model.intercept_
# coefficients (w - weights)
reg_model.coef_

##########################
# Prediction
##########################

# I have just found some coefficients. I will now proceed with the prediction process using these coefficients.

# Based on the following observation values, what is the expected value of sales?
# TV: 30
# radio: 10
# newspaper: 40

2.90794702 + 30 * 0.0468431 + 10 * 0.17854434 + 40 * 0.00258619
# 6.20213102

##########################
# Evaluating Prediction Accuracy
##########################

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# 1.736902590147092

# TRAIN R-squared
reg_model.score(X_train, y_train)
#  0.8959372632325174

# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# 1.4113417558581587

# Test R-squared
reg_model.score(X_test, y_test)
# 0.8927605914615384

# 5-fold Cross-Validated Root Mean Squared Error (RMSE)
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))
# 1.71

######################################################
# Simple Linear Regression with Gradient Descent from Scratch
######################################################

# Cost function MSE

def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0

    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2

    mse = sse / m
    return mse

# update_weights
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w

def train(Y, initial_b, initial_w, X, learning_rate, num_iters):     
    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                   cost_function(Y, initial_b, initial_w, X)))

    b = initial_b               
    w = initial_w              
    cost_history = []          

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)       
        mse = cost_function(Y, b, w, X)                      
        cost_history.append(mse)

        if i % 100 == 0:
             ("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))

    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w          

df = pd.read_csv("C:/Users/Emre Bilir/Desktop/DSBootcamp/machine_learning/datasets/advertising.csv")

X = df.drop('sales', axis=1)
y = df[["sales"]]

# hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 100000

cost_history, b, w = train(y, initial_b, initial_w, X, learning_rate, num_iters)
# Starting gradient descent at b = 0.001, w = 0.001, mse = 222.9477491673001
