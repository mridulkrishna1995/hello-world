import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# Define the functin we are trying to estimate
sin = lambda i: math.sin(2 * math.pi * i)

# Vectorize it to easily apply to numpy arrays
vsin = np.vectorize(sin)

# Generate sample data
np.random.seed(0)
X = np.random.normal(0, 1, 50)
X.sort()
X = X[:, np.newaxis]
Y = vsin(X)  # + np.random.normal(0, 0.1, X.size)

# Scatterplot to see how the sample data looks
# plt.scatter(X, Y)

# Generate test data
X_test = np.random.normal(0, 1, 20)
X_test.sort()
Y_test = vsin(X_test)
X_test = X_test[:, np.newaxis]

# Plot succesively higher degrees of polynomials trying to estimate out function
plt.scatter(X, Y)
test_errors = np.array([])
train_errors = np.array([])
for i in range(1, 4):
    # Add polynomial features of higher degree to the training and test data
    polynomial_features = PolynomialFeatures(degree=i)
    X_extend = polynomial_features.fit_transform(X)
    X_test_extend = polynomial_features.fit_transform(X_test)

    # Train a linear regression model on the training data
    model = LinearRegression()
    model.fit(X_extend, Y)

    # Compute Training error
    YPred = model.predict(X_extend)
    rmse = np.sqrt(mean_squared_error(Y, YPred))
    train_errors = np.append(train_errors, rmse)

    # Compute test error
    YPred = model.predict(X_test_extend)
    plt.plot(X_test, YPred)
    rmse = np.sqrt(mean_squared_error(Y_test, YPred))
    test_errors = np.append(test_errors, rmse)
plt.show()

plt.figure()
plt.plot(train_errors)
plt.plot(test_errors)
plt.show()
